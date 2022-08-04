from arguments import get_arguments
from networks import GridEncoder, MLP, weights_init_, get_output_shape
from overcooked_ai_py.mdp.overcooked_mdp import Action
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_dataset import OvercookedDataset, Subtasks
from overcooked_gym_env import OvercookedGymEnv
from state_encodings import ENCODING_SCHEMES

from copy import deepcopy
import numpy as np
from pathlib import Path
import pygame
from pygame.locals import HWSURFACE, DOUBLEBUF, RESIZABLE, QUIT, VIDEORESIZE
from tqdm import tqdm
import time
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical
import wandb

NUM_ACTIONS = 6 # UP, DOWN, LEFT, RIGHT, INTERACT, NOOP


class BehaviouralCloning(nn.Module):
    def __init__(self, device, visual_obs_shape, agent_obs_shape, pred_subtasks, cond_subtasks,  act=nn.ReLU, hidden_dim=256):
        """
        NN network for a behavioral cloning agent
        :param visual_obs_shape: Shape of any grid-like input to be passed into a CNN
        :param agent_obs_shape: Shape of any vector input to passed only into an MLP
        :param depth: Depth of CNN
        :param act: activation function
        :param hidden_dim: hidden dimension to use in NNs
        """
        super(BehaviouralCloning, self).__init__()
        self.device = device
        self.act = act
        self.hidden_dim = hidden_dim
        self.use_visual_obs = np.prod(visual_obs_shape) > 0
        assert len(agent_obs_shape) == 1
        self.use_agent_obs = np.prod(agent_obs_shape) > 0
        self.subtasks_obs_size = Subtasks.NUM_SUBTASKS if cond_subtasks else 0
        self.pred_subtasks = pred_subtasks
        self.cond_subtasks = cond_subtasks
        if self.cond_subtasks:
            unknown_task_id = th.tensor(Subtasks.SUBTASKS_TO_IDS['unknown']).to(self.device)
            self.curr_subtask = F.one_hot(unknown_task_id, num_classes=Subtasks.NUM_SUBTASKS)
        # Define CNN for grid-like observations
        if self.use_visual_obs:
            self.cnn = GridEncoder(visual_obs_shape)
            self.cnn_output_shape = get_output_shape(self.cnn, [1, *visual_obs_shape])[0]
        else:
            self.cnn_output_shape = 0

        # Define MLP for vector/feature based observations
        self.mlp = MLP(input_dim=self.cnn_output_shape + agent_obs_shape[0] + self.subtasks_obs_size,
                       output_dim=self.hidden_dim, hidden_dim=self.hidden_dim)
        self.action_predictor = nn.Linear(self.hidden_dim, NUM_ACTIONS)
        if self.pred_subtasks:
            self.subtask_predictor = nn.Linear(self.hidden_dim, Subtasks.NUM_SUBTASKS)
        self.apply(weights_init_)
        self.to(self.device)

    def forward(self, obs):
        visual_obs, agent_obs, subtask = obs
        latent_state = []
        # Concatenate all input features before passing them to MLP
        if self.use_visual_obs:
            # Convert all grid-like observations to features using CNN
            latent_state.append(self.cnn(visual_obs))
        if self.use_agent_obs:
            latent_state.append(agent_obs)
        if self.cond_subtasks:
            latent_state.append(subtask.to(self.device))
        latent_feats = self.mlp(th.cat(latent_state, dim=-1))
        action_logits = self.action_predictor(latent_feats)
        return (action_logits, self.subtask_predictor(latent_feats)) if self.pred_subtasks else action_logits

    def select_action(self, obs, sample=True):
        """Select action. If sample is True, sample action from distribution, else pick best scoring action"""
        logits = self.forward([th.tensor(o, device=self.device).unsqueeze(dim=0) for o in obs])
        if self.pred_subtasks:
            return (Categorical(logits=logits[0]).sample() if sample else th.argmax(logits[0], dim=-1)), logits[1]
        else:
            return Categorical(logits=logits).sample() if sample else th.argmax(logits, dim=-1)

    def predict(self, obs, sample=True):
        obs = (*obs, self.curr_subtask)
        preds = self.select_action(obs, sample=sample)
        if self.cond_subtasks:
            action = Action.INDEX_TO_ACTION[preds[0]]
            # Update predicted subtask
            if action == Action.INTERACT:
                # pred_subtask[i] = th.softmax(preds[1].detach().squeeze(), dim=-1)
                ps = th.zeros_like(preds[1].squeeze())
                ps[th.argmax(preds[1].detach().squeeze(), dim=-1)] = 1
                self.curr_subtask = ps.float()
                # print(ps.shape, th.softmax(preds[1].detach().squeeze(), dim=-1).shape)
        else:
            action = Action.INDEX_TO_ACTION[preds]
        return action, None


class BC_trainer():
    def __init__(self, encoding_fn, train_layouts, test_layout, args, vis_eval=False, pred_subtasks=True, cond_subtasks=True):
        """
        Class to train BC agent
        :param env: Overcooked environment to use
        :param encoding_fn: A callable function to encode an Overcooked state into a combination of visual and agent obs
                            that can be fed into a NN
        :param dataset: That dataset to train on - can be None if the only visualizing agetns
        :param args: arguments to use
        :param vis_eval: If true, the evaluate function will visualize the agents
        :param pred_subtasks: If true, the model will be trained to output predicted subtasks
        :param cond_subtasks: If true, the model will be conditioned on 'current' (either true or predicted) subtasks
        """
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.num_players = 2
        self.encode_state_fn = encoding_fn
        self.args = args
        self.visualize_evaluation = vis_eval
        self.pred_subtasks = pred_subtasks
        assert not (cond_subtasks and not pred_subtasks), "Can only condition on subtasks if also predicting subtasks"
        self.cond_subtasks = cond_subtasks
        self.train_layouts = train_layouts
        self.test_layout = test_layout
        self.train_dataset = OvercookedDataset(encoding_fn, self.train_layouts, args)
        self.grid_shape = self.train_dataset.grid_shape
        self.eval_env = OvercookedGymEnv(layout=test_layout, encoding_fn=encoding_fn, grid_shape=self.grid_shape, args=args)
        obs = self.eval_env.get_obs()
        visual_obs_shape = obs['visual_obs'][0].shape
        agent_obs_shape = obs['agent_obs'][0].shape
        self.players = (
            BehaviouralCloning(self.device, visual_obs_shape, agent_obs_shape, pred_subtasks, cond_subtasks),
            BehaviouralCloning(self.device, visual_obs_shape, agent_obs_shape, pred_subtasks, cond_subtasks)
        )

        if len(train_layouts) > 0:
            self.optimizers = tuple([th.optim.Adam(player.parameters(), lr=args.lr) for player in self.players])
            self.action_criterion = nn.CrossEntropyLoss(weight=th.tensor(self.train_dataset.get_action_weights(), dtype=th.float32, device=self.device))
            if self.pred_subtasks:
                self.subtask_criterion = nn.CrossEntropyLoss(weight=th.tensor(self.train_dataset.get_subtask_weights(), dtype=th.float32, device=self.device),
                                                             reduction='none')
        if self.visualize_evaluation:
            self.eval_env.setup_visualization()

    def evaluate(self, num_trials=10, sample=True):
        """
        Evaluate agent on <num_trials> trials. Returns average true reward and average shaped reward trials.
        :param num_trials: Number of trials to run
        :param sample: Boolean. If true sample from action distribution. If false, always take 'best' action.
                       NOTE: if sample is false, there is no point in running more than a single trial since the system
                             becomes deterministic
        :return: average true reward and average shaped reward
        """

        average_reward = []
        shaped_reward = []
        for trial in range(num_trials):
            self.eval_env.reset()

            unknown_task_id = th.tensor(Subtasks.SUBTASKS_TO_IDS['unknown']).to(self.device)
            # Predicted subtask to perform next, stars as unknown
            pred_subtask = [F.one_hot(unknown_task_id, num_classes=Subtasks.NUM_SUBTASKS),
                            F.one_hot(unknown_task_id, num_classes=Subtasks.NUM_SUBTASKS)]
            trial_reward, trial_shaped_r = 0, 0
            done = False
            timestep = 0
            while not done:
                # Encode Overcooked state into observations for agents
                obs = self.eval_env.get_obs()
                vis_obs = th.tensor(obs['visual_obs'], device=self.device, dtype=th.float32)
                agent_obs = th.tensor(obs['agent_obs'], device=self.device, dtype=th.float32)

                # Get next actions - we don't use overcooked gym env for this because we want to allow subtasks
                joint_action = []
                for i in range(2):
                    pi_obs = [o[i] for o in (vis_obs, agent_obs, pred_subtask)]
                    preds = self.players[i].select_action(pi_obs, sample)
                    if self.pred_subtasks:
                        action = preds[0]
                        # Update predicted subtask
                        if Action.INDEX_TO_ACTION[action] == Action.INTERACT:
                            ps = th.zeros_like(preds[1].squeeze())
                            ps[th.argmax(preds[1].detach().squeeze(), dim=-1)] = 1
                            pred_subtask[i] = ps.float()
                    else:
                        action = preds
                    joint_action.append(action)
                joint_action = tuple(joint_action)
                # Environment step
                next_state, reward, done, info = self.eval_env.step(joint_action)
                # Update metrics
                trial_reward += np.sum(info['sparse_r_by_agent'])
                trial_shaped_r += np.sum(info['shaped_r_by_agent'])
                timestep += 1
            average_reward.append(trial_reward)
            shaped_reward.append(trial_shaped_r)
        return np.mean(average_reward), np.mean(shaped_reward)

    def train_on_batch(self, batch):
        """Train BC agent on a batch of data"""
        # print({k: v for k,v in batch.items()})
        batch = {k: v.to(self.device) for k,v in batch.items()}
        vo, ao, action, subtasks = batch['visual_obs'].float(), batch['agent_obs'].float(), \
                                   batch['joint_action'].long(), batch['subtasks'].long()

        curr_subtask, next_subtask = subtasks[:, 0], subtasks[:, 1]
        metrics = {}
        for i in range(self.num_players):
            self.optimizers[i].zero_grad()
            cs_i = F.one_hot(curr_subtask[:,i], num_classes=Subtasks.NUM_SUBTASKS)
            preds = self.players[i].forward( (vo[:,i], ao[:,i], cs_i) )
            tot_loss = 0
            if self.pred_subtasks:
                # Train on subtask prediction task
                pred_action, pred_subtask = preds
                subtask_loss = self.subtask_criterion(pred_subtask, next_subtask[:, i])
                subtask_mask = action[:,i] == Action.ACTION_TO_INDEX[Action.INTERACT]
                loss_mask = th.logical_or(subtask_mask, th.rand_like(subtask_loss, device=self.device) > 0.95)
                subtask_loss = th.mean(subtask_loss * loss_mask)
                tot_loss += th.mean(subtask_loss)
                metrics[f'p{i}_subtask_loss'] = subtask_loss.item()
                pred_subtask_indices = th.argmax(pred_subtask, dim=-1)
                accuracy = ((pred_subtask_indices == next_subtask[:, i]).float() * subtask_mask).sum() / \
                           subtask_mask.float().sum()
                metrics[f'p{i}_subtask_acc'] = accuracy.item()
            else:
                pred_action = preds
            # Train on action prediction task
            action_loss = self.action_criterion(pred_action, action[:, i])
            metrics[f'p{i}_action_loss'] = action_loss.item()
            tot_loss += action_loss

            tot_loss.backward()
            self.optimizers[i].step()
        return metrics

    def train_epoch(self):
        metrics = {}
        for i in range(2):
            self.players[i].train()

        count = 0
        dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        for batch in tqdm(dataloader):
            new_losses = self.train_on_batch(batch)
            metrics = {k: [new_losses[k]] + metrics.get(k, []) for k in new_losses}
            count += 1

        metrics = {k: np.mean(v) for k, v in metrics.items()}
        metrics['total_loss'] = sum([v for k, v in metrics.items() if 'loss' in k])
        return metrics

    def training(self, exp_name, num_epochs=100):
        """ Training routine """
        run = wandb.init(project="overcooked_ai_test", entity="stephaneao", dir=str(args.base_dir / 'wandb'), reinit=True, name=exp_name)#, mode='disabled')
        best_loss = float('inf')
        best_reward = 0
        for epoch in range(num_epochs):
            mean_reward, shaped_reward = self.evaluate()
            metrics = self.train_epoch()
            wandb.log({'eval_true_reward': mean_reward, 'eval_shaped_reward': shaped_reward, 'epoch': epoch, **metrics})
            if metrics['total_loss'] < best_loss:
                print(f'Best loss achieved on epoch {epoch}, saving models')
                self.save(tag='best_loss')
                best_loss = metrics['total_loss']
            if mean_reward > best_reward:
                print(f'Best reward achieved on epoch {epoch}, saving models')
                self.save(tag='best_reward')
                best_reward = mean_reward
        run.finish()

    def save(self, tag=''):
        save_path = self.args.base_dir / 'saved_models' / f'{args.exp_name}_{tag}'
        Path(save_path).mkdir(parents=True, exist_ok=True)
        for i in range(2):
            th.save(self.players[i].state_dict(), save_path / f'player{i}')

    def load(self, load_name='default_exp_225'):
        load_path = self.args.base_dir / 'saved_models' / load_name
        for i in range(2):
            self.players[i].load_state_dict(th.load(load_path / f'player{i}', map_location=self.device))


if __name__ == '__main__':
    args = get_arguments()
    encoding_fn = ENCODING_SCHEMES[args.encoding_fn]
    eval_only = False
    if eval_only:
        bct = BC_trainer(encoding_fn, 'all', 'asymmetric_advantages', args, vis_eval=True)
        bct.load()
        bct.evaluate(10)
    else:
        # bct = BC_trainer(encoding_fn, 'all', 'asymmetric_advantages', args, vis_eval=False)
        # bct.training('all')
        # del bct
        #
        # bct = BC_trainer(encoding_fn, 'all', 'asymmetric_advantages', args, vis_eval=False, cond_subtasks=False)
        # bct.training('all_no_subtask')
        # del bct

        bct = BC_trainer(encoding_fn, ['asymmetric_advantages'], 'asymmetric_advantages', args, vis_eval=False)
        bct.training('single')
        del bct

        # bct = BC_trainer(encoding_fn, ['asymmetric_advantages'], 'asymmetric_advantages', args, vis_eval=False, cond_subtasks=False)
        # bct.training('single_no_subtask')
        # del bct
        #
        # bct = BC_trainer(encoding_fn, ['cramped_room','coordination_ring','counter_circuit','forced_coordination'], 'asymmetric_advantages', args, vis_eval=False)
        # bct.training('all_but')
        # del bct
        #
        # bct = BC_trainer(encoding_fn, ['cramped_room','coordination_ring','counter_circuit','forced_coordination'], 'asymmetric_advantages', args, vis_eval=False, cond_subtasks=False)
        # bct.training('all_but_no_subtask')
        # del bct

