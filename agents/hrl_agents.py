from agent import OAIAgent, SB3Wrapper, OAITrainer
from arguments import get_arguments
from behavioral_cloning import BehaviouralCloningPolicy, BehaviouralCloningAgent, BehavioralCloningTrainer
from overcooked_gym_env import OvercookedGymEnv
from overcooked_subtask_gym_env import OvercookedSubtaskGymEnv
from overcooked_manager_gym_env import OvercookedManagerGymEnv
from rl_agents import TwoSingleAgentsTrainer, OneDoubleAgentTrainer, SingleAgentTrainer, SB3DoubleAgentWrapper
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
           ((object.name == player.held_object.name) or
            (object.name == 'soup' and player.held_object.name == 'onion'))\
           and object.position == (x, y)


class MultiAgentSubtaskWorker(OAIAgent):
    def __init__(self, agents, p_idx, args):
        super(OAIAgent, self).__init__('multi_agent_subtask_worker', p_idx, args)
        self.agents = agents

    def predict(self, obs: th.Tensor):
        assert 'subtask' in obs.keys()
        subtask_one_hot = obs['subtask']
        subtask_id = np.nonzero(subtask_one_hot)[0][0]
        return self.agents[subtask_id].predict(obs)

    def get_distribution(self, obs: th.Tensor):
        assert 'subtask' in obs.keys()
        subtask_one_hot = obs['subtask']
        subtask_id = np.nonzero(subtask_one_hot)[0][0]
        return self.agents[subtask_id].get_distribution(obs)

    def _get_constructor_parameters(self):
        return dict(name=self.name, p_idx=self.p_idx, args=self.args)

    def save(self, path: str) -> None:
        args = get_args_to_save(self.args)
        agent_path = path + '_subtask_agents_dir'
        Path(agent_path).mkdir(parents=True, exist_ok=True)

        save_dict = {'sb3_model_type': type(self.agent), 'agent_paths': [],
                     'const_params': self._get_constructor_parameters(), 'args': args}
        for i, agent in enumerate(agents):
            agent_path_i = agent_path + '/agent{i}'
            agent.save(path)
            save_dict['agent_paths'].append(agent_path_i)
        th.save(save_dict, path)

    @classmethod
    def load(cls, path: str, args):
        device = args.device
        saved_variables = th.load(path, map_location=device)
        set_args_from_load(saved_variables['args'], args)

        # Load weights
        agents = []
        for agent_path in save_dict['agent_paths']:
            agent = saved_variables['sb3_model_type'].load(agent_path)
            agent.to(device)
            agents.append(agent)
        return cls(agents=agents, **saved_variables['const_params'], **kwargs)

    @classmethod
    def create_model_from_scratch(cls, p_idx, args, dataset_file=None) -> 'OAIAgent':
        t_idx = (p_idx + 1) % 2
        if dataset_file is not None:
            bct = BehavioralCloningTrainer(dataset_file, args)
            bct.train_agents(epochs=2)
            tm = bct.get_agent(p_idx=t_idx)
        else:
            tsa = TwoSingleAgentsTrainer(args)
            tsa.train_agents(total_timesteps=1e8)
            tm = tsa.get_agent(p_idx=t_idx)

        # Train 12 individual agents, each for a respective subtask
        agents = []
        for i in range(Subtasks.NUM_SUBTASKS):
            # RL single subtask agents trained with BC partner
            p_kwargs = {'p1': tm} if t_idx == 0 else {'p2': tm}
            kwargs = {'single_subtask_id': i, 'shape_rewards': True, 'args': args}
            env = make_vec_env(OvercookedGymEnv, n_envs=args.n_envs, env_kwargs={**p_kwargs, **env_kwargs})
            eval_env = OvercookedSubtaskGymEnv(**p_kwargs, **kwargs)
            rl_sat = SingleAgentTrainer(tm, t_idx, args, env=env, eval_env=eval_env)
            rl_sat.train_agents(total_timesteps=1e4)
            agents.append(rl_sat.get_agent(p_idx))
        path = self.args.base_dir / 'agent_models' / self.name / self.args.layout_name
        Path(path).mkdir(parents=True, exist_ok=True)
        tag = self.args.exp_name
        self.save(str(path / tag))
        return cls(agents=agents, p_idx=p_idx, args=args), tm


class Manager(OAIAgent):
    def __init__(self, worker, p_idx, args):
        super(Manager, self).__init__(f'base_manager', p_idx, args)
        self.worker = worker
        assert worker.p_idx == p_idx
        self.set_player_idx(p_idx)
        self.trajectory = []
        self.terrain = OvercookedGridworld.from_layout_name(args.layout_name).terrain_mtx
        # for i in range(len(self.terrain)):
        #     self.terrain[i] = ''.join(self.terrain[i])
        # self.terrain = str(self.terrain)
        self.worker_subtask_counts = np.zeros((2, Subtasks.NUM_SUBTASKS))
        self.encoding_fn = ENCODING_SCHEMES[args.encoding_fn]
        self.subtask_selection = args.subtask_selection
        self.reset(None)

    def select_next_subtask(self, curr_state):
        ''' Given the current state, select the next subtask to perform'''
        pass

    def predict(self, obs, sample=True):
        obs['subtask'] = self.curr_subtask_id
        action_logits = self.worker.forward(obs)
        action = Categorical(logits=action_logits).sample() if sample else th.argmax(action_logits, dim=-1)
        return action

    def get_distribution(self, obs, sample=True):
        obs['subtask'] = self.curr_subtask_id
        return self.worker.forward(obs)


    def step(self, new_state, joint_action):
        if self.p_idx is None:
            raise ValueError('Player idx must be set before Manager.step can be called')
        for i in range(2):
            if joint_action[i] == Action.ACTION_TO_INDEX[Action.INTERACT]:
                # Update subtask counts
                subtask_id = calculate_completed_subtask(self.terrain, self.curr_state, new_state, i)
                if subtask_id is not None: # i.e. If interact has an effect
                    self.worker_subtask_counts[i][subtask_id] += 1
                # Pick next subtask
                self.select_next_subtask(self.curr_state)

        self.trajectory.append(joint_action)
        if self.subtask_selection == 'value_based':
            self.update_subtask_values(self.curr_state, new_state)
        self.curr_state = new_state

    def reset(self, state):
        super().reset(state)
        self.trajectory = []
        self.curr_state = state
        # Pick first subtask
        self.select_next_subtask(self.curr_state)

class RLManagerWrapper(SB3Wrapper, Manager):
    '''
    A wrapper for a stable baselines 3 agents that controls which subtask a worker should perform.
    NOTE: When using the class loading method, it requires an additional kwarg of the worker (e.g. worker=bc_agent)
    '''
    def __init__(self, agent, worker, p_idx, args):
        # This will call SB3Wrapper init due to MRO. SB3Wrapper expects the agent passed in is the agent to save
        # There is no need to call the Manager init, since all that code is to set up trajectory collection
        super(RLManagerWrapper, self).__init__(manager, p_idx, args)
        self.name = 'rl_manager'
        self.manager = agent
        self.worker = worker

    def select_next_subtask(self, curr_state):
        obs = self.encoding_fn(curr_state)
        # obs = {k: self.obs_type(v, device=self.device) for k, v in obs.items()}
        self.curr_subtask_id = self.manager.predict(obs)[0]

    def step(self, new_state, joint_action):
        pass

    def reset(self, state):
        pass

class RLManagerTrainer(SingleAgentTrainer):
    ''' Train an RL agent to play with a provided agent '''
    def __init__(self, worker, teammate, teammate_idx, args):
        kwargs = {'worker': worker, 'teammate': teammate, 'shape_rewards': True, 'randomize_start': False, 'args': args}
        env = make_vec_env(OvercookedGymEnv, n_envs=args.n_envs, env_kwargs={**kwargs})
        eval_env = OvercookedManagerGymEnv(**kwargs)
        self.worker = worker
        super(RLManagerTrainer, self).__init__(teammate, teammate_idx, args, env=env, eval_env=eval_env)
        assert worker.p_idx == self.p_idx and teammate.p_idx == self.t_idx

    def wrap_agent(self, rl_agent):
        return RLManagerWrapper(rl_agent, self.worker, self.p_idx, self.args)

class ValueBasedManager(Manager):
    """
    Follows a few basic rules. (tm = teammate)
    1. All independent tasks values:
       a) Get onion from dispenser = (3 * num_pots) - 0.5 * num_onions
       b) Get plate from dish rack = num_filled_pots * (2
       c)
       d)
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
    def __init__(self, agent, p_idx, args):
        super(ValueBasedManager, self).__init__(agent, p_idx, args)
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
        assert self.subtask_values is not None
        return self.subtask_values * self.doable_subtasks(curr_state, self.terrain, self.p_idx)

    def select_next_subtask(self, curr_state):
        subtask_values = self.get_subtask_values(curr_state)
        subtask_id = np.argmax(subtask_values.squeeze(), dim=-1)
        self.curr_subtask_id = subtask_id
        print('new subtask', Subtasks.IDS_TO_SUBTASKS[subtask_id.item()])

    def reset(self, state):
        super().reset(state)
        self.init_subtask_values()

class DistBasedManager(Manager):
    def __init__(self, agent, p_idx, args):
        super(DistBasedManager, self).__init__(agent, p_idx, args)
        self.name = 'dist_based_subtask_agent'

    def distribution_matching(self, subtask_logits, egocentric=False):
        """
        Try to match some precalculated 'optimal' distribution of subtasks.
        If egocentric look only at the individual player distribution, else look at the distribution across both players
        """
        assert self.optimal_distribution is not None
        if egocentric:
            curr_dist = self.worker_subtask_counts[self.p_idx]
            best_dist = self.optimal_distribution[self.p_idx]
        else:
            curr_dist = self.worker_subtask_counts.sum(axis=0)
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

if __name__ == '__main__':
    args = get_arguments()
    p_idx, t_idx = 0, 1
    worker, teammate = MultiAgentSubtaskWorker.create_model_from_scratch(p_idx, args, dataset_file=args.dataset)

    #create_rl_worker(args.dataset, p_idx, args)
    # tsat = TwoSingleAgentsTrainer(args)
    # tsat.train_agents(total_timesteps=1e6)
    # teammate = tsat.get_agent(t_idx)
    rlmt = RLManagerTrainer(worker, teammate, t_idx, args)
    rlmt.train_agents(total_timesteps=1e8)


