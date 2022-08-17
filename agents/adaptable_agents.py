from agent import OAIAgent
from behavioral_cloning import BehaviouralCloningPolicy, BehaviouralCloningAgent, BehavioralCloningTrainer
from rl_agents import TwoSingleAgentsTrainer, OneDoubleAgentTrainer

from subtasks import Subtasks, calculate_completed_subtask

import numpy as np

class SubtaskAdaptor(BehaviouralCloningAgent):
    def __init__(self, device, visual_obs_shape, agent_obs_shape, args):
        super(BehaviouralCloningAgent, self).__init__(device, visual_obs_shape, agent_obs_shape, args)
        self.subtask_adaptor = None # TODO
        self.player_idx = None
        self.trajectory = []
        self.terrain = OvercookedGridworld.from_layout_name(args.layout_name).terrain_mtx
        self.agent_subtask_counts = np.zeros( (2, Subtasks.NUM_SUBTASKS) )
        self.encoding_fn = ENCODING_SCHEMES[args.encoding_fn]

    def subtask_adaptor_tabular_rl(self, subtask_logits):
        """Use tabular RL on subtasks"""
        pass

    def subtask_adaptor_distribution_matching(self):
        """Try to match some preset 'optimal' distribution of subtasks"""
        pass

    def subtask_adaptor_simple_rl(self):
        """Upweight supporting tasks if humans perform the complementary task"""
        pass

    def update_subtask_logits(self, subtask_logits):
        # TODO update logits based on some adaptation
        pass

    def predict(self, obs, sample=True):
        obs = (*obs, self.curr_subtask) if self.use_subtasks else obs
        obs = [o.unsqueeze(dim=0).to(self.device) for o in obs]
        logits = self.forward(obs)
        if self.use_subtasks:
            action_logits, subtask_logits = logits
            # Update predicted subtask
            if Action.INDEX_TO_ACTION[action] == Action.INTERACT:
                adapted_subtask_logits = self.subtask_adaptor(subtask_logits)
                ps = th.zeros_like(adapted_subtask_logits.squeeze())
                ps[th.argmax(adapted_subtask_logits.detach().squeeze(), dim=-1)] = 1
                self.adapted_subtask_logits = ps.float()
        else:
            action_logits = logits
        return Categorical(logits=action_logits).sample() if sample else th.argmax(action_logits, dim=-1)

    def step(self, new_state, joint_action):
        if self.player_idx is None:
            raise ValueError('Player idx must be set before SubtaskAdaptor.step can be called')
        interact_id = Action.ACTION_TO_INDEX[Action.INTERACT]
        for i in range(2):
            if joint_action[i] == interact_id:
                subtask_id = calculate_completed_subtask(self.terrain, self.curr_state, new_state, i)
                self.agent_subtask_counts[i][subtask_id] += 1

        self.trajectory.append(joint_action)

    def reset(self, state, player_idx):
        self.trajectory = []
        self.curr_state = state
        # Predicted subtask to perform next, starts as unknown
        unknown_task_id = th.tensor(Subtasks.SUBTASKS_TO_IDS['unknown']).to(self.device)
        self.curr_subtask = F.one_hot(unknown_task_id, num_classes=Subtasks.NUM_SUBTASKS)


    def save(self, path: str):
        """All models should be pretrained, so saving shouldn't be necessary"""
        pass

    def load(self, path: str):
        """All models should be pretrained and passed in already loaded, so loading shouldn't be necessary"""
        pass

class TypeBasedAdaptor(OAIAgent):
    def __init__(self, args):
        super(TypeBasedAdaptor, self).__init__()
        self.args = args

        self.policy = np.random.choice(self.models)
        self.env = OvercookedGymEnv(p1_agent=p1, p2_agent=p2, layout=layout, encoding_fn=self.encoding_fn, args=args)
        self.encoding_fn = self.env.encoding_fn

    def create_models(self):
        # TODO add options to each kind of agent (e.g. reward shaping / A2C vs PPO for RL agents, using subtasks for BC Agents
        self.p1_models = {}
        self.p2_models = {}
        # RL double agent
        rl_odat = OneDoubleAgentTrainer(self.args)
        rl_odat.train_agents(epochs=1000)
        self.p1_models['rl_double_agent_p1'] = rl_odat.get_agent(idx=0)
        self.p2_models['rl_double_agent_p2'] = rl_odat.get_agent(idx=1)
        # RL single agents
        rl_tsat = TwoSingleAgentsTrainer(self.args)
        rl_tsat.train_agents(epochs=1000)
        self.p1_models['rl_single_agent_p1'] = rl_tsat.get_agent(idx=0)
        self.p2_models['rl_single_agent_p2'] = rl_tsat.get_agent(idx=1)
        # BC agents
        bct = BehavioralCloningTrainer(self.args)
        bct.train_agents(epochs=1000)
        self.p1_models['bc_agent_p1'] = bct.get_agent(idx=0)
        self.p2_models['bc_agent_p2'] = bct.get_agent(idx=1)
        # TODO RL Agent trained with BC agent
        # TODO deal with different layouts logic

        self.selfplay_table = self.calculate_selfplay_table(args.layout_name)
        self.save()


    def predict(self, obs: th.Tensor) -> Tuple[int, Union[th.Tensor, None]]:
        """
        Given an observation return the index of the action and the agent state if the agent is recurrent.
        """
        return self.policy.predict(obs)

    def step(self, state, joint_action):
        if self.player_idx is None:
            raise ValueError('Player idx must be set before TypeBasedAdaptor.step can be called')
        self.trajectory.append((state, joint_action))
        teammate_action = joint_action[self.player_idx]
        self.update_policy_dist(state, teammate_action)

    def reset(self, state, player_idx):
        # NOTE this is for a reset between episodes. Create a new TypeBasedAdaptor if this is for a new human
        super(TypeBasedAdaptor, self).reset(state, player_idx)
        self.policy = self.select_policy()
        self.trajectory = []

    def update_policy_dist(self, state, action):
        # TODO updates the estimated probability of which model best resembles teammate
        pass

    def calculate_selfplay_table(self, layout):
        # TODO calculate the score for each model playing with each other
        selfplay_table = pd.DataFrame(columns=list(self.p1_models.keys()), index=list(self.p2_models.keys()))
        for p1_name, p1 in self.p1_models.items():
            for p2_name, p2 in self.p2_models.item():
                eval_env = OvercookedGymEnv(p1_agent=p1, p2_agent=p2, layout=layout, encoding_fn=self.encoding_fn, args=args)
                selfplay_table[p1_name][p2_name] = eval_env.run_full_episode()
        return selfplay_table

    def select_policy_using_cross_entropy_metric(self, trajectory, horizon=10):
        prior_teammate_policies, t_idx = (self.p2_models, 1) if self.player_idx == 0 else (self.p1_models, 0)
        horizon = min(horizon, len(trajectory))
        trajectory = trajectory[-horizon:]
        best_cem = 0
        chosen_policy = None
        for policy in prior_teammate_policies:
            cem = 0
            for t in range(horizon):
                state, joint_action = trajectory[t]
                action = joint_action[t_idx]
                obs = self.encoding_fn(self.env.mdp, state, self.env.grid_shape, self.args.horizon, p_idx=t_idx)
                dist = policy.get_distribution(obs)
                cem += dist.log_prob(action)
            cem = cem / horizon
            if cem > best_cem:
                chosen_policy = policy
                best_cem = cem
        return chosen_policy

    def select_policy(self):
        # TODO use distribution to find the most similar model to human,
        #      then select the most complementary model using the selfplay_table
        if self.use_cem:
            self.curr_policy = self.select_policy_using_cross_entropy_metric(self.trajectory)

    def save(self, path: str=None):
        """Save all models and selfplay table"""
        base_dir = args.base_dir / 'agent_models' / 'type_based_agents' / self.args.layout_name
        for model_name, model in self.p1_models.items():
            save_path = base_dir / model_name
            model.save(save_path)

        for model_name, model in self.p2_models.items():
            save_path = base_dir / model_name
            model.save(save_path)

        self.selfplay_table.to_pickle(base_dir / 'selfplay_table.pickle')


    def load(self, path: str=None):
        """Load all models and selfplay table"""
        base_dir = args.base_dir / 'agent_models' / 'type_based_agents' / self.args.layout_name
        for model_name, model in self.p1_models.items():
            load_path = base_dir / model_name
            model.load(load_path)

        for model_name, model in self.p2_models.items():
            load_path = base_dir / model_name
            model.load(load_path)

        self.selfplay_table.read_pickle(base_dir / 'selfplay_table.pickle')


# TODO wandb init each agent at the start of their training