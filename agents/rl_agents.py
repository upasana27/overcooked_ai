import gym
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
import torch as th
import wandb

from arguments import get_arguments
from agent import OAIAgent, SB3Wrapper, OAITrainer
from networks import OAISinglePlayerFeatureExtractor, OAISinglePlayerLSTMFeatureExtractor, OAIDoublePlayerFeatureExtractor
from overcooked_gym_env import OvercookedGymEnv
from overcooked_subtask_gym_env import OvercookedSubtaskGymEnv
from state_encodings import ENCODING_SCHEMES


class SB3SingleAgentWrapper(SB3Wrapper):
    ''' A wrapper for a stable baselines 3 agents that controls a single player '''
    def predict(self, obs):
        return self.agent.predict(obs)

    def get_distribution(self, obs: th.Tensor):
        return self.agent.get_distribution(obs)


class SB3DoubleAgentWrapper(SB3Wrapper):
    ''' A wrapper for a stable baselines agents that controls both players (i.e. outputs an action for both) '''
    def predict(self, obs):
        action, state = self.agent.predict(obs)
        return action[self.p_idx], state

    def get_distribution(self, obs: th.Tensor):
        return self.agent.get_distribution(obs)[self.p_idx]


class SingleAgentTrainer(OAITrainer):
    ''' Train an RL agent to play with a provided agent '''
    def __init__(self, teammates, teammate_idx, args, env=None, use_lstm=False, hidden_dim=256, seed=seed):
        super(SingleAgentTrainer, self).__init__('single_agent', args)
        self.args = args
        self.device = args.device
        self.encoding_fn = ENCODING_SCHEMES[args.encoding_fn]
        self.teammates = teammates
        self.curr_teammate_idx = 0
        teammate = teammates[self.curr_teammate_idx]
        self.t_idx, self.p_idx = teammate_idx, (teammate_idx + 1) % 2
        p_kwargs = {'p1': teammate} if teammate_idx == 0 else {'p2': teammate}
        kwargs = {'shape_rewards': True, 'return_traj_id': use_lstm, 'args': args}
        self.env = env or OvercookedGymEnv(**p_kwargs, **kwargs)

        fec = OAISinglePlayerLSTMFeatureExtractor if use_lstm else OAISinglePlayerFeatureExtractor
        policy_kwargs = dict(
            features_extractor_class=fec,
            features_extractor_kwargs=dict(features_dim=hidden_dim),
            net_arch=[dict(pi=[hidden_dim, hidden_dim], vf=[hidden_dim, hidden_dim])]
        )
        sb3_agent = PPO("MultiInputPolicy", self.env, policy_kwargs=policy_kwargs, verbose=1)
        sb3_agent.policy.to(self.device)
        learner = self.wrap_agent(sb3_agent)
        self.agents = [teammate, learner] if teammate_idx == 0 else [learner, teammate]
        for agent in self.agents:
            agent.policy.to(self.device)

    def wrap_agent(self, sb3_agent):
        return SB3SingleAgentWrapper(sb3_agent, f'rl_single_agent_p{self.p_idx + 1}', self.p_idx, self.args)

    def train_agents(self, epochs=1000, exp_name=None):
        exp_name = exp_name or self.args.exp_name
        run = wandb.init(project="overcooked_ai_test", entity=self.args.wandb_ent, dir=str(self.args.base_dir / 'wandb'),
                         reinit=True, name=exp_name + '_' + self.agents[self.p_idx].name, mode=self.args.wandb_mode)
        for i in range(2):
            self.agents[i].policy.train()
        best_score, scores = 0, []
        best_path, best_tag = None, None
        for epoch in range(epochs):
            if epoch % 10 == 0:
                score, done_training = self.env.evaluate(self.agents[self.p_idx], num_trials=1)
                scores.append(score)
                print(f'Episode eval at epoch {epoch}: {score}')
                wandb.log({'eval_true_score': score, 'epoch': epoch})
                if score > best_score:
                    best_path, best_tag = self.save()
                    best_score = score
                if len(scores) > 5 and score <= np.mean(scores[-4:-1]): # no improvement
                    break
            self.agents[self.p_idx].learn(total_timesteps=1000)
            self.curr_teammate_idx = (self.curr_teammate_idx + 1) % len(self.teammates)
            self.env.set_agent(self.teammates[self.curr_teammate_idx], self.t_idx)

        if best_path is not None:
            self.load(best_path, best_tag)
        run.finish()

    def get_agent(self, idx):
        if idx != self.p_idx:
            raise ValueError(f'This trainer only trained a player {self.p_idx + 1} agent, '
                             f'and therefore cannot return a {self.t_idx + 1} agent')
        return self.agents[self.p_idx]


class TwoSingleAgentsTrainer(OAITrainer):
    ''' Train two independent RL agents to play with each other '''
    def __init__(self, args, use_lstm=False, hidden_dim=256, seed=seed, fcp_ck_rate=None):
        super(TwoSingleAgentsTrainer, self).__init__('two_single_agents', args)
        self.device = args.device
        self.args = args
        self.encoding_fn = ENCODING_SCHEMES[args.encoding_fn]
        self.save_fcp_ck = save_fcp_ck
        kwargs = {'shape_rewards': True, 'return_traj_id': use_lstm, 'args': args}
        self.envs = [OvercookedGymEnv(p2='temp', **kwargs), OvercookedGymEnv(p1='temp', **kwargs)]

        fec = OAISinglePlayerLSTMFeatureExtractor if use_lstm else OAISinglePlayerFeatureExtractor
        policy_kwargs = dict(
            features_extractor_class=fec,
            features_extractor_kwargs=dict(features_dim=hidden_dim),
            net_arch=[dict(pi=[hidden_dim, hidden_dim], vf=[hidden_dim, hidden_dim])]
        )
        agents = [PPO("MultiInputPolicy", self.envs[0], policy_kwargs=policy_kwargs, verbose=1),
                  PPO("MultiInputPolicy", self.envs[1], policy_kwargs=policy_kwargs, verbose=1)]
        self.agents = [SB3SingleAgentWrapper(agents[i], f'rl_two_single_agents_p{i + 1}', i, args) for i in range(2)]

        self.eval_env = OvercookedGymEnv(p1=self.agents[0], p2=self.agents[1], return_traj_id=use_lstm, args=args)
        self.envs[0].set_agent(self.agents[1], idx=1)
        self.envs[1].set_agent(self.agents[0], idx=0)
        for agent in self.agents:
            agent.policy.to(self.device)

    def train_agents(self, epochs=1000, exp_name=None):
        if self.save_fcp_ck is not None:
            save_dict = {}
        exp_name = exp_name or self.args.exp_name
        run = wandb.init(project="overcooked_ai_test", entity=self.args.wandb_ent, dir=str(self.args.base_dir / 'wandb'),
                         reinit=True, name=exp_name + '_rl_two_single_agents', mode=self.args.wandb_mode)
        for i in range(2):
            self.agents[i].policy.train()
        best_score, scores = 0, []
        best_path, best_tag = None, None
        for epoch in range(epochs):
            self.agents[0].learn(total_timesteps=1000)
            print('done learning 1')
            self.agents[1].learn(total_timesteps=1000)
            print('done learning 2')
            if epoch % 10 == 0:
                score = self.eval_env.run_full_episode()
                scores.append(score)
                print(f'Episode eval at epoch {epoch}: {score}')
                wandb.log({'eval_true_reward': score, 'epoch': epoch})
                if score > best_score:
                    best_path, best_tag = self.save()
                    best_score = score
                if len(scores) > 5 and score <= np.mean(scores[-4:-1]): # done training
                    break
                if self.save_fcp_ck is not None and epoch % self.save_fcp_ck == 0:
                    self.save(tag=f'epoch_{epoch}')

        if best_path is not None:
            self.load(best_path, best_tag)
        run.finish()

    def get_agent(self, p_idx):
        return self.agents[p_idx]


class OneDoubleAgentTrainer(OAITrainer):
    ''' Train an RL agent to play both agents '''
    def __init__(self, args, hidden_dim=256, seed=seed):
        super(OneDoubleAgentTrainer, self).__init__('one_double_agent', args)
        self.device = args.device
        self.args = args
        self.env = OvercookedGymEnv(shape_rewards=True, args=args)
        policy_kwargs = dict(
            features_extractor_class=OAIDoublePlayerFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=hidden_dim),
            net_arch=[dict(pi=[hidden_dim, hidden_dim], vf=[hidden_dim, hidden_dim])]
        )
        self.agent = PPO("MultiInputPolicy", self.env, policy_kwargs=policy_kwargs, verbose=1)
        self.agents = [SB3DoubleAgentWrapper(self.agent, f'rl_one_double_agent_p{i + 1}', i, args) for i in range(2)]
        self.eval_env = OvercookedGymEnv(args=args)
        for agent in self.agents:
            agent.policy.to(self.device)

    def train_agents(self, epochs=1000, exp_name=None):
        exp_name = exp_name or self.args.exp_name
        run = wandb.init(project="overcooked_ai_test", entity=self.args.wandb_ent, dir=str(self.args.base_dir / 'wandb'),
                         reinit=True, name=exp_name + '_rl_double_agent', mode=self.args.wandb_mode)
        self.agent.policy.train()
        best_score, scores = 0, []
        best_path, best_tag = None, None
        for epoch in range(epochs):
            self.agent.learn(total_timesteps=10000)
            if epoch % 10 == 0:
                score = self.run_full_episode()
                scores.append(score)
                print(f'Episode eval at epoch {epoch}: {score}')
                wandb.log({'eval_true_reward': score, 'epoch': epoch})
                if score > best_score:
                    best_path, best_tag = self.save()
                    best_score = score
                if len(scores) > 5 and score <= np.mean(scores[-4:-1]): # done training
                    break
        if best_path is not None:
            self.load(best_path, best_tag)
        run.finish()

    def run_full_episode(self):
        self.eval_env.reset()
        done = False
        total_reward = 0
        while not done:
            joint_action, _ = self.agent.predict(self.eval_env.get_obs())
            obs, reward, done, info = self.eval_env.step(joint_action)
            total_reward += np.sum(info['sparse_r_by_agent'])
        return total_reward

    def get_agent(self, idx):
        agent = SB3DoubleAgentWrapper(self.agent, idx)
        agent.set_name(f'rl_double_agent_p{idx + 1}')
        return agent

def create_fcp_teammates(t_idx, args):
    for use_lstm in [True, False]:
        for hidden_dim in [16, 256]:
            for seed in np.random.randint(0, 1e8):
                rl_sat = TwoSingleAgentsTrainer(args, use_lstm=use_lstm, hidden_dim=hidden_dim, seed=seed)
                rl_sat.train_agents(epochs=2000)
                new_agent = rl_sat.get_agent(t_idx)
    return worker


if __name__ == '__main__':
    args = get_arguments()
    lstm = TwoSingleAgentsTrainer(args, use_lstm=True)
    lstm.train_agents(epochs=25)
    tsa = TwoSingleAgentsTrainer(args)
    tsa.train_agents(epochs=25)
    oda = OneDoubleAgentTrainer(args)
    oda.train_agents(epochs=200)




