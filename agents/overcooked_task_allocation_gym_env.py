from overcooked_gym_env import OvercookedGymEnv
from subtasks import Subtasks, get_doable_subtasks, calculate_completed_subtask

import numpy as np


class OvercookedTaskAllocationGymEnv(OvercookedGymEnv):
    def __init__(self, subtask_agent=None, teammate=None, grid_shape=None, shape_rewards=False, obs_type=None, randomize_start=False, args=None):
        assert subtask_agent.p_idx != teammate.p_idx
        self.subtask_agent_idx = subtask_agent.p_idx
        p1, p2 = (subtask_agent, teammate) if subtask_agent.p_idx == 0 else (teammate, subtask_agent)
        args.horizon = 1200
        super(OvercookedTaskAllocationGymEnv, self).__init__(p1, p2, grid_shape, shape_rewards, obs_type, randomize_start, args)
        assert any(self.agents) and self.p_idx is not None
        self.observation_space = spaces.Dict({
            "visual_obs": spaces.Box(0, 20, obs['visual_obs'].shape, dtype=np.int),
            "agent_obs": spaces.Box(0, self.args.horizon, obs['agent_obs'].shape, dtype=np.float32),
        })

    def get_obs(self, p_idx=None):
        obs = self.encoding_fn(self.env.mdp, self.state, self.grid_shape, self.args.horizon, p_idx=p_idx)
        if self.obs_type == th.tensor:
            return {k: self.obs_type(v, device=self.device) for k, v in obs.items()}
        else:
            return {k: self.obs_type(v) for k, v in obs.items()}

    def step(self, action):
        # Action is the subtask for subtask agent to perform

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
        next_state, reward, done, info = self.env.step(joint_action)
        self.state = self.env.state

        if Action.INDEX_TO_ACTION[joint_action[self.p_idx]] == Action.INTERACT:
            subtask = calculate_completed_subtask(self.mdp.terrain_mtx, self.prev_state, self.state, self.p_idx)
            if subtask is not None:
                done = True
                reward = 1 if subtask == self.goal_subtask else -1

        return self.get_obs(self.p_idx), reward, done, info

    def reset(self):
        self.env.reset()
        self.state = self.env.state
        subtask_mask = get_doable_subtasks(self.state, self.mdp.terrain_mtx, self.p_idx)
        self.goal_subtask = np.random.choice(Subtasks.SUBTASKS, p=subtask_mask)
        self.goal_subtask_id = th.tensor(Subtasks.SUBTASKS_TO_IDS[self.goal_subtask]).to(self.device)
        self.goal_subtask_one_hot = F.one_hot(self.goal_subtask_id, num_classes=Subtasks.NUM_SUBTASKS)
        for i in range(2):
            if isinstance(self.agents[i], OAIAgent):
                self.agents[i].reset(self.state, i)
        return self.get_obs(self.p_idx)

    def evaluate(self, main_agent=None, num_trials=25, other_agent=None):
        assert main_agent is not None and other_agent is None
        results = np.zeros((Subtasks.NUM_SUBTASKS, 2))
        for trial in num_trials:
            reward, done = 0, False
            obs = self.reset()
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


