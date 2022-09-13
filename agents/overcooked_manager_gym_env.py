from agent import OAIAgent
from overcooked_gym_env import OvercookedGymEnv
from subtasks import Subtasks, get_doable_subtasks, calculate_completed_subtask
from state_encodings import ENCODING_SCHEMES

from copy import deepcopy
from gym import Env, spaces, make, register
import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import Action
import torch as th
import torch.nn.functional as F


class OvercookedManagerGymEnv(OvercookedGymEnv):
    def __init__(self, worker=None, teammate=None, grid_shape=None, shape_rewards=False, randomize_start=False, args=None):
        assert worker.p_idx != teammate.p_idx
        self.worker = worker
        self.worker_idx = worker.p_idx
        p1, p2 = (None, teammate) if worker.p_idx == 0 else (teammate, None)
        super(OvercookedManagerGymEnv, self).__init__(p1=p1, p2=p2, grid_shape=grid_shape, shape_rewards=shape_rewards, randomize_start=randomize_start, args=args)
        assert any(self.agents) and self.p_idx is not None
        self.action_space = spaces.Discrete(Subtasks.NUM_SUBTASKS)

    def get_obs(self, p_idx=None):
        obs = self.encoding_fn(self.env.mdp, self.state, self.grid_shape, self.args.horizon, p_idx=p_idx)
        if p_idx == self.worker_idx:
            obs['subtask'] = [self.curr_subtask]
        return obs

    def step(self, action):
        # Action is the subtask for subtask agent to perform
        if type(action) == th.tensor:
            self.curr_subtask = action.cpu()
        joint_action = [Action.STAY, Action.STAY]
        reward, done, info = 0, False, None
        while joint_action[self.worker_idx] != Action.INTERACT and not done:
            joint_action[self.p_idx] = self.worker.predict(self.get_obs(p_idx=self.p_idx))[0][0]
            joint_action[self.t_idx] = self.agents[self.t_idx].predict(self.get_obs(p_idx=self.t_idx))[0][0]
            # joint_action = [self.agents[i].predict(self.get_obs(p_idx=i))[0] for i in range(2)]
            joint_action = [Action.INDEX_TO_ACTION[a] for a in joint_action]

            # If the state didn't change from the previous timestep and the agent is choosing the same action
            # then play a random action instead. Prevents agents from getting stuck
            if self.state.time_independent_equal(self.prev_state) and tuple(joint_action) == self.prev_actions:
                joint_action = [np.random.choice(Action.ALL_ACTIONS), np.random.choice(Action.ALL_ACTIONS)]

            self.prev_state, self.prev_actions = deepcopy(self.state), joint_action
            for i in range(2):
                if isinstance(self.agents[i], OAIAgent):
                    self.agents[i].step(self.prev_state, joint_action)
            next_state, r, done, info = self.env.step(joint_action)
            reward += r
            self.state = self.env.state

        return self.get_obs(self.p_idx), reward, done, info

    def reset(self):
        self.env.reset()
        self.state = self.env.state
        for i in range(2):
            if isinstance(self.agents[i], OAIAgent):
                self.agents[i].reset(self.state)
        self.curr_subtask = 0
        return self.get_obs(self.p_idx)

    def run_full_episode(self):
        assert self.agents[0] is not None and self.agents[1] is not None
        obs = self.reset()
        for i in range(2):
            self.agents[i].policy.eval()
        done = False
        total_reward = 0
        while not done:
            if self.visualization_enabled:
                self.render()
            action = self.agents[i].predict(obs)[0]
            obs, reward, done, info = self.step(action)

            total_reward += np.sum(info['sparse_r_by_agent'])
        return total_reward
