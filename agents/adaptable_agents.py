from agent import OAIAgent
from behavioral_cloning import BehaviouralCloningPolicy, BehaviouralCloningAgent, BehavioralCloningTrainer
from overcooked_gym_env import OvercookedGymEnv
from rl_agents import TwoSingleAgentsTrainer, OneDoubleAgentTrainer, SingleAgentTrainer, DoubleAgentWrapper
from subtasks import Subtasks, calculate_completed_subtask, get_doable_subtasks
from state_encodings import ENCODING_SCHEMES

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.actions import Action

import numpy as np
import torch as th
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from typing import Tuple, Union


# TODO Move to util
def is_held_obj(player, object):
    '''Returns True if the object that the "player" picked up / put down is the same as the "object"'''
    x, y = np.array(player.position) + np.array(player.orientation)
    return player.held_object is not None and \
           (
                (object.name == player.held_object.name) or
                (object.name == 'soup' and player.held_object.name == 'onion')
           )\
           and object.position == (x, y)



class SubtaskAdaptor(BehaviouralCloningAgent):
    def __init__(self, visual_obs_shape, agent_obs_shape, idx, args):
        args.use_subtasks = True
        super(SubtaskAdaptor, self).__init__(visual_obs_shape, agent_obs_shape, idx, args)
        self.subtask_adaptor = None  # TODO
        self.trajectory = []
        self.terrain = OvercookedGridworld.from_layout_name(args.layout_name).terrain_mtx
        # for i in range(len(self.terrain)):
        #     self.terrain[i] = ''.join(self.terrain[i])
        # self.terrain = str(self.terrain)
        self.agent_subtask_counts = np.zeros((2, Subtasks.NUM_SUBTASKS))
        self.encoding_fn = ENCODING_SCHEMES[args.encoding_fn]
        self.subtask_selection = args.subtask_selection
        if self.subtask_selection == 'weighted':
            self.init_subtask_weighting()
        self.name = f'subtask_adaptor_{self.subtask_selection}'
        self.reset(None, idx)

    def subtask_adaptor_tabular_rl(self, subtask_logits):
        """
        Use tabular RL on subtasks
        """
        pass

    def subtask_adaptor_distribution_matching(self, subtask_logits, egocentric=False):
        """
        Try to match some precalculated 'optimal' distribution of subtasks.
        If egocentric look only at the individual player distribution, else look at the distribution across both players
        """
        assert self.optimal_distribution is not None

        if egocentric:
            curr_dist = self.agent_subtask_counts[self.p_idx]
            best_dist = self.optimal_distribution[self.p_idx]
        else:
            curr_dist = self.agent_subtask_counts.sum(axis=0)
            best_dist = self.optimal_distribution.sum(axis=0)
        curr_dist = curr_dist / np.sum(curr_dist)
        dist_diff = best_dist - curr_dist

        pred_subtask_probs = F.softmax(subtask_logits).detach().numpy()
        # TODO investigate weighting
        # TODO should i do the above softmax?
        # Loosely based on Bayesian inference where prior is the difference in distributions, and the evidence is
        # predicted probability of what subtask should be done
        adapted_probs = pred_subtask_probs * dist_diff
        adapted_probs = adapted_probs / np.sum(adapted_probs)
        return adapted_probs

    def init_subtask_weighting(self):
        self.subtask_logits_weights = np.zeros(Subtasks.NUM_SUBTASKS)
        # 'unknown' subtask is always set to 0 since it is more a relic of labelling than a useful subtask
        # Independent subtasks
        self.ind_subtask = ['get_onion_from_dispenser', 'put_onion_in_pot', 'get_plate_from_dish_rack', 'get_soup', 'serve_soup']
        # Supportive subtasks
        self.sup_subtask = ['put_onion_closer', 'put_plate_closer', 'put_soup_closer']
        self.sup_obj_to_subtask = {'onion': 'put_onion_closer', 'dish': 'put_plate_closer', 'soup': 'put_soup_closer'}
        # Complementary subtasks
        self.com_subtask = ['get_onion_from_counter', 'get_plate_from_counter', 'get_soup_from_counter']
        self.com_obj_to_subtask = {'onion': 'get_onion_from_counter', 'dish': 'get_plate_from_counter', 'soup': 'get_soup_from_counter'}
        for i_s in self.ind_subtask:
            # 1.a
            self.subtask_logits_weights[Subtasks.SUBTASKS_TO_IDS[i_s]] = 1
        for s_s in self.sup_subtask:
            # 2.a
            self.subtask_logits_weights[Subtasks.SUBTASKS_TO_IDS[s_s]] = 0
        for c_s in self.com_subtask:
            # 3.a
            self.subtask_logits_weights[Subtasks.SUBTASKS_TO_IDS[c_s]] = 0

        self.acceptable_wait_time = 10  # 2d
        self.sup_base_inc = 0.05  # 2b
        self.sup_success_inc = 1  # 2c
        self.sup_waiting_dec = 0.1  # 2d
        self.com_waiting_inc = 0.2  # 3d
        self.successful_support_task_reward = 1
        self.agent_objects = {}
        self.teammate_objects = {}

    def update_subtask_weighting(self, prev_state, curr_state):
        prev_objects = prev_state.objects.values()
        curr_objects = curr_state.objects.values()
        # TODO objects are only tracked by name and position, so checking equality fails because picking something up changes the objects position
        # 2.b
        for s_s in self.sup_subtask:
            self.subtask_logits_weights[Subtasks.SUBTASKS_TO_IDS[s_s]] += self.sup_base_inc

        # Analyze objects that are on counters
        for object in curr_objects:
            x, y = object.position
            if object.name == 'soup' and self.terrain[y][x] == 'P':
                # Soups while in pots can change without agent intervention
                continue
            # Objects that have been put down this turn
            if object not in prev_objects:
                if is_held_obj(prev_state.players[self.p_idx], object):
                    print(f'Agent placed {object}')
                    self.agent_objects[object] = 0
                elif is_held_obj(prev_state.players[self.t_idx], object):
                    print(f'Teammate placed {object}')
                    self.teammate_objects[object] = 0
                # else:
                #     raise ValueError(f'Object {object} has been put down, but did not belong to either player')
            # Objects that have not moved since the previous time step
            else:
                if object in self.agent_objects:
                    self.agent_objects[object] += 1
                    if self.agent_objects[object] > self.acceptable_wait_time:
                        # 2.d
                        subtask_id = Subtasks.SUBTASKS_TO_IDS[self.sup_obj_to_subtask[object.name]]
                        self.subtask_logits_weights[subtask_id] -= self.sup_waiting_dec
                elif object in self.teammate_objects:
                    # 3.b
                    self.teammate_objects[object] += 1
                    subtask_id = Subtasks.SUBTASKS_TO_IDS[self.com_obj_to_subtask[object.name]]
                    self.subtask_logits_weights[subtask_id] += self.com_waiting_inc

        for object in prev_objects:
            x, y = object.position
            if object.name == 'soup' and self.terrain[y][x] == 'P':
                # Soups while in pots can change without agent intervention
                continue
            # Objects that have been picked up this turn
            if object not in curr_objects:
                if is_held_obj(curr_state.players[self.p_idx], object):
                    print(f'Agent picked up {object}')
                    if object in self.agent_objects:
                        del self.agent_objects[object]
                    else:
                        del self.teammate_objects[object]

                elif is_held_obj(curr_state.players[self.t_idx], object):
                    print(f'Teammate picked up {object}')
                    if object in self.agent_objects:
                        # 2.c
                        subtask_id = Subtasks.SUBTASKS_TO_IDS[self.sup_obj_to_subtask[object.name]]
                        self.subtask_logits_weights[subtask_id] += self.sup_success_inc
                        del self.agent_objects[object]
                    else:
                        del self.teammate_objects[object]
                # else:
                #     raise ValueError(f'Object {object} has been picked up, but does not belong to either player')

                # Find out if there are any remaining objects of the same type left
                last_object_of_this_type = True
                for rem_objects in list(self.agent_objects) + list(self.teammate_objects):
                    if object.name == rem_objects.name:
                        last_object_of_this_type = False
                        break
                # 3.c
                if last_object_of_this_type:
                    subtask_id = Subtasks.SUBTASKS_TO_IDS[self.com_obj_to_subtask[object.name]]
                    self.subtask_logits_weights[subtask_id] = 0

        self.subtask_logits_weights = np.clip(self.subtask_logits_weights, 0, 10)

    def subtask_adaptor_subtask_weighting(self, subtask_logits, curr_state):
        """
        Follows a few basic rules. (tm = teammate)
        1. All independent tasks:
           a) Always have a weight of one.
        2. Supporting tasks
           a) Start at a weight of zero
           b) Always increases in weight by a small amount (supporting is good)
           c) If one is performed and the tm completes the complementary task, then the task weight is increased
           d) If the tm doesn't complete the complementary task, after a grace period the task weight starts decreasing
              until the object is picked up
        3. Complementary tasks:
           a) Start at a weight of zero
           b) If a tm performs a supporting task, then its complementary task weight increases while the object remains
              on the counter.
           c) If the object is removed from the counter, the complementary task weight is reset to zero (the
              complementary task cannot be completed if there is no object to pick up)
        :return:
        """
        assert self.subtask_logits_weights is not None
        subtask_probs = F.softmax(subtask_logits).detach()
        return subtask_probs * self.subtask_logits_weights * self.doable_subtasks(curr_state, self.terrain, self.p_idx)

    def update_subtask_logits(self, subtask_logits, curr_state):
        # TODO update logits based on some adaptation
        if self.subtask_selection == 'weighted':
            subtask_logits = self.subtask_adaptor_subtask_weighting(subtask_logits, curr_state)
            return subtask_logits

    def predict(self, obs, sample=True):
        obs = [(th.tensor(o, dtype=th.float32).to(self.device) if not isinstance(o, th.Tensor) else o.to(self.device))
               for o in [obs['visual_obs'], obs['agent_obs']]]
        obs = [o.unsqueeze(dim=0).to(self.device) for o in (*obs, self.curr_subtask)]
        action_logits, self.subtask_logits = self.forward(obs)
        action = Categorical(logits=action_logits).sample() if sample else th.argmax(action_logits, dim=-1)
        return action

    def step(self, new_state, joint_action):
        if self.p_idx is None:
            raise ValueError('Player idx must be set before SubtaskAdaptor.step can be called')
        for i in range(2):
            if joint_action[i] == Action.ACTION_TO_INDEX[Action.INTERACT] or self.first_step:
                # Update subtask counts
                subtask_id = calculate_completed_subtask(self.terrain, self.curr_state, new_state, i)
                if subtask_id is not None: # i.e. If interact is acted with no effect
                    self.agent_subtask_counts[i][subtask_id] += 1
                # Update predicted subtask
                adapted_subtask_logits = self.update_subtask_logits(self.subtask_logits, new_state)
                subtask_id = th.argmax(adapted_subtask_logits.squeeze(), dim=-1)
                ps = th.zeros_like(adapted_subtask_logits.squeeze())
                ps[subtask_id] = 1
                self.adapted_subtask_logits = ps.float()
                self.first_step = False
                print('new subtask', Subtasks.IDS_TO_SUBTASKS[subtask_id.item()])

        self.trajectory.append(joint_action)
        if self.subtask_selection == 'weighted':
            self.update_subtask_weighting(self.curr_state, new_state)
        self.curr_state = new_state

    def reset(self, state, player_idx):
        super().reset(state, player_idx)
        self.trajectory = []
        self.curr_state = state
        # Predicted subtask to perform next, starts as unknown until the first step
        unknown_task_id = th.tensor(Subtasks.SUBTASKS_TO_IDS['unknown']).to(self.device)
        self.curr_subtask = F.one_hot(unknown_task_id, num_classes=Subtasks.NUM_SUBTASKS)
        self.first_step = True
        self.subtask_logits = th.ones(Subtasks.NUM_SUBTASKS)


class TypeBasedAdaptor(OAIAgent):
    def __init__(self, p1_agents, p2_agents, selfplay_table, idx, args):
        super(TypeBasedAdaptor, self).__init__()
        self.args = args
        self.set_player_idx(idx)
        self.p1_agents = p1_agents
        self.p2_agents = p2_agents
        self.selfplay_table = selfplay_table
        self.encoding_fn = ENCODING_SCHEMES[args.encoding_fn]
        self.policy_selection = args.policy_selection
        self.name = f'type_based_adaptor_{self.policy_selection}_p{idx + 1}'
        self.env = OvercookedGymEnv(p1=p1_agents[0], p2=p2_agents[0], args=args)
        self.policy = np.random.choice(self.p1_agents if idx == 0 else self.p2_agents)


    @staticmethod
    def create_models(args, bc_epochs=2, rl_epochs=2):
        # TODO add options to each kind of agent (e.g. reward shaping / A2C vs PPO for RL agents, using subtasks for BC Agents
        p1_agents = []
        p2_agents = []

        # BC agents
        for dataset_file in ['tf_test_5_5.1.pickle', 'tf_test_5_5.2.pickle', 'tf_test_5_5.3.pickle',
                             'tf_test_5_5.4.pickle', 'tf_test_5_5.5.pickle', 'all_trials.pickle']:
            bct = BehavioralCloningTrainer(dataset_file, args)
            bce = bc_epochs #(bc_epochs // 5) if dataset_file == 'all_trials.pickle' else bc_epochs
            bct.train_agents(epochs=bce)
            bc_p1 = bct.get_agent(idx=0)
            bc_p2 = bct.get_agent(idx=1)
            p1_agents.append(bc_p1)
            p2_agents.append(bc_p2)
        
        # RL two single agents
        rl_tsat = TwoSingleAgentsTrainer(args)
        rl_tsat.train_agents(epochs=rl_epochs)
        p1_agents.append(rl_tsat.get_agent(idx=0))
        p2_agents.append(rl_tsat.get_agent(idx=1))

        # RL double agent
        rl_odat = OneDoubleAgentTrainer(args)
        rl_odat.train_agents(epochs=rl_epochs)
        p1_agents.append(rl_odat.get_agent(idx=0))
        p2_agents.append(rl_odat.get_agent(idx=1))

        # RL single agents trained with BC partner
        rl_sat = SingleAgentTrainer(bc_p2, 1, args)
        rl_sat.train_agents(epochs=rl_epochs)
        p1_agents.append(rl_sat.get_agent(idx=0))
        rl_sat = SingleAgentTrainer(bc_p1, 0, args)
        rl_sat.train_agents(epochs=rl_epochs)
        p2_agents.append(rl_sat.get_agent(idx=1))

        # TODO deal with different layouts logic
        selfplay_table = TypeBasedAdaptor.calculate_selfplay_table(p1_agents, p2_agents, args)
        return p1_agents, p2_agents, selfplay_table
        
    @staticmethod
    def calculate_selfplay_table(p1_agents, p2_agents, args):
        selfplay_table = np.zeros((len(p1_agents), len(p2_agents)))
        for i, p1 in enumerate(p1_agents):
            for j, p2 in enumerate(p2_agents):
                selfplay_table[i, j] = TypeBasedAdaptor.run_full_episode((p1, p2), args)
        print(selfplay_table)
        return selfplay_table

    @staticmethod
    def run_full_episode(players, args):
        args.horizon = 1200
        env = OvercookedGymEnv(args=args)
        env.reset()
        for player in players:
            player.policy.eval()

        done = False
        total_reward = 0
        while not done:
            if env.visualization_enabled:
                env.render()
            joint_action = [None, None]
            for i, player in enumerate(players):
                if isinstance(player, DoubleAgentWrapper):
                    joint_action[i] = player.predict(env.get_obs(p_idx=None))[0]
                else:
                    joint_action[i] = player.predict(env.get_obs(p_idx=i))[0]
            obs, reward, done, info = env.step(joint_action)

            total_reward += np.sum(info['sparse_r_by_agent'])
        return total_reward


    def predict(self, obs: th.Tensor) -> Tuple[int, Union[th.Tensor, None]]:
        """
        Given an observation return the index of the action and the agent state if the agent is recurrent.
        """
        return self.policy.predict(obs)

    def get_distribution(self, obs: th.Tensor):
        return self.policy.get_distribution(obs)

    def step(self, state, joint_action):
        if self.p_idx is None:
            raise ValueError('Player idx must be set before TypeBasedAdaptor.step can be called')
        self.trajectory.append((state, joint_action))
        self.update_beliefs(state, joint_action)

    def reset(self, state, player_idx):
        # NOTE this is for a reset between episodes. Create a new TypeBasedAdaptor if this is for a new human
        super().reset(state, player_idx)
        self.policy = self.select_policy()
        self.init_behavior_dist()
        self.trajectory = []

    def init_behavior_dist(self):
        # initialize probability distribution as uniform
        self.eta = self.args.eta
        assert len(self.p1_agents) == len(self.p2_agents)
        n = len(self.p1_agents)
        self.behavior_dist = np.full((2, n), 1 / n)

    def update_beliefs(self, state, joint_action):
        # Based on algorithm 4 (PLASTIC-Policy) of https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/AIJ16-Barrett.pdf
        prior_teammate_policies = self.p1_agents if self.t_idx == 0 else self.p2_agents
        action = joint_action[self.t_idx]
        for i, policy in enumerate(prior_teammate_policies):
            action_idx = Action.ACTION_TO_INDEX[action]
            obs = self.encoding_fn(self.env.mdp, state, self.env.grid_shape, self.args.horizon, p_idx=self.t_idx)
            prob_dist = model.get_distribution(obs)
            action_prob = prob_dist[action_idx]
            # calculate loss for model
            loss_model = 1 - action_prob
            # Update Probability Distribution according to loss for that model
            self.behavior_dist[self.t_idx][i] *= (1 - self.eta * loss_model)
        # Normalize Probabiity Distribution
        self.behavior_dist[self.t_idx] = self.behavior_dist[self.t_idx] / np.sum(self.behavior_dist[self.t_idx])

    def select_policy_plastic(self):
        best_match_policy_idx = np.argmax(self.behavior_dist[self.t_idx])
        return self.get_best_complementary_policy(best_match_policy_idx)

    def select_policy_using_cross_entropy_metric(self, horizon=10):
        # teammate_idx === t_idx
        prior_teammate_policies = self.p1_agents if self.t_idx == 0 else self.p2_agents
        horizon = min(horizon, len(self.trajectory))
        trajectory = self.trajectory[-horizon:]
        best_cem = 0
        best_match_policy_idx = None
        for policy_idx, policy in enumerate(prior_teammate_policies):
            cem = 0
            for t in range(horizon):
                state, joint_action = trajectory[t]
                action_idx = Action.ACTION_TO_INDEX[joint_action[self.t_idx]]
                obs = self.encoding_fn(self.env.mdp, state, self.env.grid_shape, self.args.horizon, p_idx=self.t_idx)
                dist = policy.get_distribution(obs)
                cem += dist.log_prob(action_idx)
            cem = cem / horizon
            if cem > best_cem:
                best_match_policy_idx = policy_idx
                best_cem = cem

        return self.get_best_complementary_policy(best_match_policy_idx)

    def get_best_complementary_policy(self, policy_idx):
        # NOTE if t_idx == 0 then p_idx == 1 and vice versa
        team_scores = self.selfplay_table[policy_idx, :] if self.t_idx == 0 else self.selfplay_table[:, policy_idx]
        own_policies = self.p1_agents if self.p_idx == 0 else self.p2_agents
        return own_policies[np.argmax(team_scores)]

    def select_policy(self):
        # TODO use distribution to find the most similar model to human,
        #      then select the most complementary model using the selfplay_table
        if self.policy_selection == 'CEM':
            self.curr_policy = self.select_policy_using_cross_entropy_metric()
        elif self.policy_selection == 'PLASTIC':
            self.curr_policy = self.select_policy_plastic()
        else:
            raise NotImplementedError(f'{self.policy_selection} is not an implemented policy selection algorithm')
        print(f'Now using policy {self.curr_policy.name}.')

    def save(self, path: str = None):
        """Save all models and selfplay table"""
        base_dir = args.base_dir / 'agent_models' / 'type_based_agents' / self.args.layout_name
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        for model in self.p1_agents:
            save_path = base_dir / model.name
            model.save(save_path)

        for model in self.p2_agents:
            save_path = base_dir / model.name
            model.save(save_path)

        self.selfplay_table.to_pickle(base_dir / 'selfplay_table.pickle')

    def load(self, path: str = None):
        """Load all models and selfplay table"""
        base_dir = args.base_dir / 'agent_models' / 'type_based_agents' / self.args.layout_name
        for model in self.p1_agents:
            load_path = base_dir / model.name
            model.load(load_path)

        for model in self.p2_agents:
            load_path = base_dir / model.name
            model.load(load_path)

        self.selfplay_table.read_pickle(base_dir / 'selfplay_table.pickle')

# TODO wandb init each agent at the start of their training


if __name__ == '__main__':
    from arguments import get_arguments
    args = get_arguments()
    p1_agents, p2_agents, sp_table = TypeBasedAdaptor.create_models(args)
    tba = TypeBasedAdaptor(p1_agents, p2_agents, sp_table, 0, args)
