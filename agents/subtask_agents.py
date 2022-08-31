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



class SubtaskAgent(OAIAgent):
    def __init__(self, agent, p_idx, args):
        super(SubtaskAgent, self).__init__()
        self.base_agent = agent
        assert agent.p_idx == p_idx
        self.set_player_idx(p_idx)
        self.subtask_adaptor = None  # TODO
        self.trajectory = []
        self.terrain = OvercookedGridworld.from_layout_name(args.layout_name).terrain_mtx
        # for i in range(len(self.terrain)):
        #     self.terrain[i] = ''.join(self.terrain[i])
        # self.terrain = str(self.terrain)
        self.agent_subtask_counts = np.zeros((2, Subtasks.NUM_SUBTASKS))
        self.encoding_fn = ENCODING_SCHEMES[args.encoding_fn]
        self.subtask_selection = args.subtask_selection
        self.name = f'base_subtask_adaptor'
        self.reset(None, idx)

    @staticmethod
    def create_subtask_adaptor_with_rl_base_agent(dataset_file, p_idx, args):
        bct = BehavioralCloningTrainer(dataset_file, args)
        bct.train_agents(epochs=1000)
        t_idx = (p_idx + 1) % 2
        bc_agent = bct.get_agent(idx=t_idx)

        # RL single subtask agents trained with BC partner
        rl_sat = SingleAgentTrainer(bc_agent, t_idx, args, use_subtask_env=True)
        rl_sat.train_agents(epochs=1000)
        subtask_agent = rl_sat.get_agent(p_idx)
        return SubtaskAgent(subtask_agent, p_idx, args)

    @staticmethod
    def create_subtask_adaptor_with_bc_base_agent(dataset_file, p_idx, args):
        args.use_subtasks = True
        bct = BehavioralCloningTrainer(dataset_file, args)
        bct.train_agents(epochs=1000)
        subtask_agent = bct.get_agent(idx=p_idx)
        return SubtaskAgent(subtask_agent, p_idx, args)

    def select_next_subtask(self, curr_state):
        ''' Given the current state, select the next subtask to perform'''
        pass

    def predict(self, obs, sample=True):
        obs = [(th.tensor(o, dtype=th.float32).to(self.device) if not isinstance(o, th.Tensor) else o.to(self.device))
               for o in [obs['visual_obs'], obs['agent_obs']]]
        obs = [o.unsqueeze(dim=0).to(self.device) for o in (*obs, self.curr_subtask)]
        action_logits = self.base_agent.forward(obs)
        action = Categorical(logits=action_logits).sample() if sample else th.argmax(action_logits, dim=-1)
        return action

    def step(self, new_state, joint_action):
        if self.p_idx is None:
            raise ValueError('Player idx must be set before SubtaskAgent.step can be called')
        for i in range(2):
            if joint_action[i] == Action.ACTION_TO_INDEX[Action.INTERACT]:
                # Update subtask counts
                subtask_id = calculate_completed_subtask(self.terrain, self.curr_state, new_state, i)
                if subtask_id is not None: # i.e. If interact has an effect
                    self.agent_subtask_counts[i][subtask_id] += 1
                # Pick next subtask
                self.select_next_subtask(self.curr_state)

        self.trajectory.append(joint_action)
        if self.subtask_selection == 'value_based':
            self.update_subtask_values(self.curr_state, new_state)
        self.curr_state = new_state

    def reset(self, state, player_idx):
        super().reset(state, player_idx)
        self.trajectory = []
        self.curr_state = state
        # Pick first subtask
        self.select_next_subtask(self.curr_state)

class RLSubtaskAgent(SubtaskAgent):
    def __init__(self, agent, p_idx, args):
        super(RLSubtaskAgent, self).__init__(agent, p_idx, args)
        self.name = 'rl_subtask_agent'

    def select_next_subtask(self, curr_state):
        # TODO
        pass

class RLSubtaskAgentTrainer(OAITrainer):
    ''' Train an RL agent to play with a provided agent '''
    def __init__(self, teammate, teammate_idx, args, use_subtask_env=False):
        super(RLSubtaskAgentTrainer, self).__init__(args)
        self.device = args.device
        self.args = args
        self.encoding_fn = ENCODING_SCHEMES[args.encoding_fn]
        self.t_idx = teammate_idx
        self.p_idx = (teammate_idx + 1) % 2
        kwargs = {'shape_rewards': True, 'obs_type': th.tensor, 'args': args}
        self.env = # TODO create new gym env to train this agent where an action is selecting the next subtask

        policy_kwargs = dict(
            features_extractor_class=OAISinglePlayerFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )
        self.agents = [teammate, PPO("MultiInputPolicy", self.env, policy_kwargs=policy_kwargs, verbose=1)] \
                      if teammate_idx == 0 else \
                      [PPO("MultiInputPolicy", self.env, policy_kwargs=policy_kwargs, verbose=1), teammate]
        self.agents[0].policy.to(device)
        self.agents[1].policy.to(device)

    def train_agents(self, epochs=1000, exp_name=None):
        exp_name = exp_name or self.args.exp_name
        run = wandb.init(project="overcooked_ai_test", entity=self.args.wandb_ent, dir=str(self.args.base_dir / 'wandb'),
                         reinit=True, name=exp_name + '_rl_single_agent', mode=self.args.wandb_mode)
        for i in range(2):
            self.agents[i].policy.train()
        best_cum_rew = 0
        best_path, best_tag = None, None
        for epoch in range(epochs):
            self.agents[self.p_idx].learn(total_timesteps=10000)
            if epoch % 10 == 0:
                cum_rew = self.env.evaluate(self.agents[self.p_idx], num_trials=1)
                print(f'Episode eval at epoch {epoch}: {cum_rew}')
                wandb.log({'eval_true_reward': cum_rew, 'epoch': epoch})
                if cum_rew > best_cum_rew:
                    best_path, best_tag = self.save()
                    best_cum_rew = cum_rew
        if best_path is not None:
            self.load(best_path, best_tag)
        run.finish()

    def get_agent(self, idx):
        if idx != self.p_idx:
            raise ValueError(f'This trainer only trained a player {self.p_idx + 1} agent, '
                             f'and therefore cannot return a {self.t_idx + 1} agent')
        agent = SingleAgentWrapper(self.agents[idx], idx)
        agent.set_name(f'rl_single_agent_with_bc_p{idx + 1}')
        return agent

    def save(self, path=None, tag=None):
        # TODO

    def load(self, path=None, tag=None):
        # TODO


class ValueBasedSubtaskAgent(SubtaskAgent):
    def __init__(self, agent, p_idx, args):
        super(ValueBasedSubtaskAgent, self).__init__(agent, p_idx, args)
        self.name = 'value_based_subtask_adaptor'
        self.init_subtask_values()

    def init_subtask_values(self):
        self.subtask_values = np.zeros(Subtasks.NUM_SUBTASKS)
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
            self.subtask_values[Subtasks.SUBTASKS_TO_IDS[i_s]] = 1
        for s_s in self.sup_subtask:
            # 2.a
            self.subtask_values[Subtasks.SUBTASKS_TO_IDS[s_s]] = 0
        for c_s in self.com_subtask:
            # 3.a
            self.subtask_values[Subtasks.SUBTASKS_TO_IDS[c_s]] = 0

        self.acceptable_wait_time = 10  # 2d
        self.sup_base_inc = 0.05  # 2b
        self.sup_success_inc = 1  # 2c
        self.sup_waiting_dec = 0.1  # 2d
        self.com_waiting_inc = 0.2  # 3d
        self.successful_support_task_reward = 1
        self.agent_objects = {}
        self.teammate_objects = {}

    def update_subtask_values(self, prev_state, curr_state):
        prev_objects = prev_state.objects.values()
        curr_objects = curr_state.objects.values()
        # TODO objects are only tracked by name and position, so checking equality fails because picking something up changes the objects position
        # 2.b
        for s_s in self.sup_subtask:
            self.subtask_values[Subtasks.SUBTASKS_TO_IDS[s_s]] += self.sup_base_inc

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
                        self.subtask_values[subtask_id] -= self.sup_waiting_dec
                elif object in self.teammate_objects:
                    # 3.b
                    self.teammate_objects[object] += 1
                    subtask_id = Subtasks.SUBTASKS_TO_IDS[self.com_obj_to_subtask[object.name]]
                    self.subtask_values[subtask_id] += self.com_waiting_inc

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
                        self.subtask_values[subtask_id] += self.sup_success_inc
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
                    self.subtask_values[subtask_id] = 0

        self.subtask_values = np.clip(self.subtask_values, 0, 10)

    def get_subtask_values(self, curr_state):
        """
        Follows a few basic rules. (tm = teammate)
        1. All independent tasks:
           a) Always have a value of one.
        2. Supporting tasks
           a) Start at a value of zero
           b) Always increases in value by a small amount (supporting is good)
           c) If one is performed and the tm completes the complementary task, then the task value is increased
           d) If the tm doesn't complete the complementary task, after a grace period the task value starts decreasing
              until the object is picked up
        3. Complementary tasks:
           a) Start at a value of zero
           b) If a tm performs a supporting task, then its complementary task value increases while the object remains
              on the counter.
           c) If the object is removed from the counter, the complementary task value is reset to zero (the
              complementary task cannot be completed if there is no object to pick up)
        :return:
        """
        assert self.subtask_values is not None
        return self.subtask_values * self.doable_subtasks(curr_state, self.terrain, self.p_idx)

    def select_next_subtask(self, curr_state):
        subtask_values = self.get_subtask_values(curr_state)
        subtask_id = th.argmax(subtask_values.squeeze(), dim=-1)
        self.curr_subtask = F.one_hot(subtask_id, Subtasks.NUM_SUBTASKS).to(self.device)
        print('new subtask', Subtasks.IDS_TO_SUBTASKS[subtask_id.item()])

    def reset(self, state, player_idx):
        super().reset(state, player_idx)
        self.init_subtask_values()

class DistBasedSubtaskAgent(SubtaskAgent):
    def __init__(self, agent, p_idx, args):
        super(DistBasedSubtaskAgent, self).__init__(agent, p_idx, args)
        self.name = 'dist_based_subtask_agent'

    def distribution_matching(self, subtask_logits, egocentric=False):
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

    def select_next_subtask(self, curr_state):
        # TODO
        pass

    def reset(self, state, player_idx):
        # TODO
        pass



