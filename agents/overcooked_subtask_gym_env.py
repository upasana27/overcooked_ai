from agent import OAIAgent
from overcooked_gym_env import OvercookedGymEnv
from subtasks import Subtasks, get_doable_subtasks, calculate_completed_subtask

from copy import deepcopy
from gym import Env, spaces, make, register
import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import Action, Direction
import torch as th
import torch.nn.functional as F


class OvercookedSubtaskGymEnv(OvercookedGymEnv):
    def __init__(self, p1=None, p2=None, grid_shape=None, shape_rewards=False, obs_type=None, randomize_start=True,
                 use_curriculum=True, args=None):
        self.use_curriculum = use_curriculum
        if self.use_curriculum:
            self.curr_lvl = 0
        else:
            self.curr_lvl = Subtasks.NUM_SUBTASKS
        super(OvercookedSubtaskGymEnv, self).__init__(p1, p2, grid_shape, shape_rewards, obs_type, randomize_start,
                                                      100, args)
        assert any(self.agents) and self.p_idx is not None
        self.observation_space = spaces.Dict({
            "visual_obs": spaces.Box(0, 20, self.visual_obs_shape, dtype=np.int),
            "agent_obs": spaces.Box(0, self.args.horizon, self.agent_obs_shape, dtype=np.float32),
            "subtask": spaces.Box(0, 1, (Subtasks.NUM_SUBTASKS,), dtype=np.int32)
        })

    def get_obs(self, p_idx=None):
        obs = self.encoding_fn(self.env.mdp, self.state, self.grid_shape, self.args.horizon, p_idx=p_idx)
        if p_idx == self.p_idx:
            obs['subtask'] = self.goal_subtask_one_hot
        if self.obs_type == th.tensor:
            return {k: self.obs_type(v, device=self.device) for k, v in obs.items()}
        else:
            return {k: self.obs_type(v) for k, v in obs.items()}

    def get_proximity_reward(self, feature_locations):
        # Calculate reward for using the pass.
        # Reward should be proportional to how much time is saved from using the pass
        smallest_dist = float('inf')
        object_location = np.array(self.state.players[self.p_idx].position) + np.array(self.state.players[self.p_idx].orientation)
        for direction in [np.array([0, 1]), np.array([0, -1]), np.array([1, 0]), np.array([-1, 0])]:
            adj_tile = tuple(np.array(object_location) + direction)
            # Can't pick up from a terrain location that is not walkable
            if adj_tile not in self.mdp.get_valid_player_positions():
                continue
            if adj_tile == self.state.players[self.p_idx].position:
                curr_dist = self.mlam.motion_planner.min_cost_to_feature((adj_tile, Direction.NORTH), feature_locations)
            else:
                dist = self.mlam.motion_planner.min_cost_to_feature((adj_tile, Direction.NORTH), feature_locations)
                if dist < smallest_dist:
                    smallest_dist = dist
        smallest_dist = min(smallest_dist, curr_dist + 1)
        # Reward proportional to how much time is saved from using the pass
        return (curr_dist - smallest_dist) * 0.1

    def step(self, action):
        joint_action = [(self.agents[i].predict(self.get_obs(p_idx=i))[0] if self.agents[i] else action) for i in range(2)]
        joint_action = [Action.INDEX_TO_ACTION[a] for a in joint_action]

        # If the state didn't change from the previous timestep and the agent is choosing the same action
        # then play a random action instead. Prevents agents from getting stuck
        if self.state.time_independent_equal(self.prev_state) and tuple(joint_action) == self.prev_actions:
            joint_action = [np.random.choice(Action.ALL_ACTIONS), np.random.choice(Action.ALL_ACTIONS)]

        self.prev_state, self.prev_actions = deepcopy(self.state), joint_action
        for i in range(2):
            if isinstance(self.agents[i], OAIAgent):
                self.agents[i].step(self.prev_state, joint_action)
        next_state, _, done, info = self.env.step(joint_action)
        self.state = self.env.state

        reward = -0.0001 # existence penalty
        if joint_action[self.p_idx] == Action.INTERACT:
            subtask = calculate_completed_subtask(self.mdp.terrain_mtx, self.prev_state, self.state, self.p_idx)
            if subtask is not None:
                done = True
                reward = 1 if subtask == self.goal_subtask_id else -0.1
                if self.goal_subtask == 'put_onion_closer':
                    pot_locations = self.mdp.get_pot_locations()
                    reward += self.get_proximity_reward(pot_locations)
                elif self.goal_subtask == 'put_plate_closer':
                    pot_locations = self.mdp.get_pot_locations()
                    reward += self.get_proximity_reward(pot_locations)
                elif self.goal_subtask == 'put_soup_closer':
                    serving_locations = self.mdp.get_serving_locations()
                    reward += self.get_proximity_reward(serving_locations)

        return self.get_obs(self.p_idx), reward, done, info

    def reset(self, evaluate=False):
        doable_subtask_in_curr_lvl = False
        while not doable_subtask_in_curr_lvl:
            self.env.reset()
            self.state = self.env.state
            subtask_mask = get_doable_subtasks(self.state, self.mdp.terrain_mtx, self.p_idx)
            doable_subtask_in_curr_lvl = np.nonzero(subtask_mask)[0][0] <= self.curr_lvl
        # JUST FOR TESTING ###
        # print(f'doable subtasks for {self.p_idx}:')
        # for subtask_id, doable in enumerate(subtask_mask):
        #     if doable:
        #         print(Subtasks.IDS_TO_SUBTASKS[subtask_id])
        # self.render()
        # input('Hit enter to start')
        ###
        # nothing past curr level can be selected
        subtask_mask[self.curr_lvl + 1:] = 0
        subtask_probs = subtask_mask / np.sum(subtask_mask)
        self.goal_subtask = np.random.choice(Subtasks.SUBTASKS, p=subtask_probs)
        self.goal_subtask_id = th.tensor(Subtasks.SUBTASKS_TO_IDS[self.goal_subtask]).to(self.device)
        self.goal_subtask_one_hot = F.one_hot(self.goal_subtask_id, num_classes=Subtasks.NUM_SUBTASKS)
        for i in range(2):
            if isinstance(self.agents[i], OAIAgent):
                self.agents[i].reset(self.state)
        return self.get_obs(self.p_idx)

    def evaluate(self, main_agent=None, num_trials=25, other_agent=None):
        assert main_agent is not None and other_agent is None
        results = np.zeros((Subtasks.NUM_SUBTASKS, 2))
        num_trials = 25
        for trial in range(num_trials):
            reward, done = 0, False
            obs = self.reset(evaluate=True)
            while not done:
                action = main_agent.predict(obs)[0]
                obs, reward, done, info = self.step(action)

            if reward == 1:
                results[self.goal_subtask_id][0] += 1
            else:
                results[self.goal_subtask_id][1] += 1

        for subtask in Subtasks.SUBTASKS:
            subtask_id = Subtasks.SUBTASKS_TO_IDS[subtask]
            print(f'{subtask_id} - successes: {results[subtask_id][0]}, failures: {results[subtask_id][1]}')
        if np.sum(results[:, 0]) == num_trials and self.curr_lvl < Subtasks.NUM_SUBTASKS:
            print(f'Going from level {self.curr_lvl} to {self.curr_lvl + 1}')
            self.curr_lvl += 1
        return np.sum(results[:, 0])


