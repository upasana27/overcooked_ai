from agent import OAIAgent
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, OvercookedGridworld, Direction, Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from state_encodings import ENCODING_SCHEMES

from copy import deepcopy
from gym import Env, spaces, make, register
import numpy as np
import pygame
from pygame.locals import HWSURFACE, DOUBLEBUF, RESIZABLE, QUIT, VIDEORESIZE
from stable_baselines3.common.env_checker import check_env
import torch as th


class OvercookedGymEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, p1=None, p2=None, grid_shape=None, shape_rewards=False, base_env=None, horizon=None, args=None):
        self.args = args
        self.device = args.device
        self.encoding_fn = ENCODING_SCHEMES[args.encoding_fn]
        self.agents = [p1, p2]
        if base_env is None:
            self.mdp = OvercookedGridworld.from_layout_name(args.layout_name)
            horizon = horizon or args.horizon
            self.env = OvercookedEnv.from_mdp(self.mdp, horizon=horizon)
        else:
            self.env = base_env
        self.visualization_enabled = False
        self.grid_shape = grid_shape or self.env.mdp.shape
        self.shape_rewards = shape_rewards
        self.step_count = 0
        # If we play one agent, we play the agent that is not defined, otherwise both agents are equal
        self.p_idx = (1 if self.agents[0] else 0) if any(self.agents) else None
        self.t_idx = (self.p_idx + 1) % 2 if self.p_idx is not None else None
        obs = self.reset()
        self.visual_obs_shape = obs['visual_obs'].shape if 'visual_obs' in obs else 0
        self.agent_obs_shape = obs['agent_obs'].shape if 'agent_obs' in obs else 0
        # TODO improve bounds for each dimension
        # Currently 20 is the default value for recipe time (which I believe is the largest value used
        self.obs_dict = {}
        if np.prod(self.visual_obs_shape) > 0:
            self.obs_dict["visual_obs"] = spaces.Box(0, 20, self.visual_obs_shape, dtype=np.int)
        if np.prod(self.agent_obs_shape) > 0:
            self.obs_dict["agent_obs"] =  spaces.Box(0, self.args.horizon, self.agent_obs_shape, dtype=np.float32)
        self.observation_space = spaces.Dict(self.obs_dict)

        self.prev_state, self.prev_actions = deepcopy(self.state), (Action.STAY, Action.STAY)
        if all(self.agents):  # We control no agents
            self.action_space = spaces.Space()
        elif any(self.agents):  # We control 1 agent
            self.action_space = spaces.Discrete(len(Action.ALL_ACTIONS))
        else:  # We control both agents
            self.action_space = spaces.MultiDiscrete([ len(Action.ALL_ACTIONS), len(Action.ALL_ACTIONS) ])
        

    def set_agent(self, agent, idx):
        self.agents[idx] = agent

    def setup_visualization(self):
        self.visualization_enabled = True
        pygame.init()
        surface = StateVisualizer().render_state(self.state, grid=self.env.mdp.terrain_mtx)
        self.window = pygame.display.set_mode(surface.get_size(), HWSURFACE | DOUBLEBUF | RESIZABLE)
        self.window.blit(surface, (0, 0))
        pygame.display.flip()

    def get_obs(self, p_idx=None):
        obs = self.encoding_fn(self.env.mdp, self.state, self.grid_shape, self.args.horizon, p_idx=p_idx)
        return obs

    def step(self, action):
        if all(self.agents): # We control no agents
            joint_action = [self.agents[i].predict(self.get_obs(p_idx=i))[0] for i in range(2)]
        elif any(self.agents): # We control 1 agent
            joint_action = [(self.agents[i].predict(self.get_obs(p_idx=i))[0] if self.agents[i] else action)
                            for i in range(2)]
        else: # We control both agents
            joint_action = action

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
        if self.shape_rewards:
            ratio = min(self.step_count / 2.5e6, 1)
            sparse_r = sum(info['sparse_r_by_agent'])
            shaped_r = info['shaped_r_by_agent'][self.p_idx] if self.p_idx else sum(info['shaped_r_by_agent'])
            reward = sparse_r * ratio + shaped_r * (1 - ratio)
        self.step_count += 1
        return self.get_obs(self.p_idx), reward, done, info

    def reset(self):
        self.env.reset()
        self.state = self.env.state
        for i in range(2):
            if isinstance(self.agents[i], OAIAgent):
                self.agents[i].reset(self.state)
        return self.get_obs(self.p_idx)

    def render(self, mode='human', close=False):
        if self.visualization_enabled:
            surface = StateVisualizer().render_state(self.state, grid=self.env.mdp.terrain_mtx)
            self.window = pygame.display.set_mode(surface.get_size(), HWSURFACE | DOUBLEBUF | RESIZABLE)
            self.window.blit(surface, (0, 0))
            pygame.display.flip()
            pygame.time.wait(100)

    def close(self):
        pygame.quit()

    def evaluate(self, main_agent=None, num_trials=25, other_agent=None):
        reset_p1_to_none, reset_p2_to_none = False, False
        if all(self.agents):
            pass
        elif any(self.agents):
            if self.agents[0] is None:
                assert main_agent.p_idx == 0
                self.agents[0] = main_agent
                reset_p1_to_none = True
            else:
                assert main_agent.p_idx == 1
                self.agents[1] = main_agent
                reset_p2_to_none = True
        else:
            assert main_agent.p_idx != other_agent.p_idx
            self.agents[main_agent.p_idx] = main_agent
            self.agents[(main_agent + 1) % 2] = other_agent
            reset_p1_to_none, reset_p2_to_none = True, True

        rewards = []
        for trial in range(num_trials):
            rewards.append(self.run_full_episode())
        print(f'Mean reward = {np.mean(rewards)}')
        if reset_p1_to_none:
            self.agents[0] = None
        if reset_p2_to_none:
            self.agents[1] = None
        return np.mean(rewards), False


    def run_full_episode(self):
        assert self.agents[0] is not None and self.agents[1] is not None
        self.reset()
        for i in range(2):
            self.agents[i].policy.eval()
        done = False
        total_reward = 0
        while not done:
            if self.visualization_enabled:
                self.render()
            obs, reward, done, info = self.step(None)

            total_reward += np.sum(info['sparse_r_by_agent'])
        return total_reward

register(
    id='OvercookedGymEnv-v0',
    entry_point='OvercookedGymEnv'
)

class DummyAgent:
    def __init__(self, action=Action.STAY):
        self.action = Action.ACTION_TO_INDEX[action]

    def predict(self, x):
        return self.action, None

if __name__ == '__main__':
    from state_encodings import encode_state
    from arguments import get_arguments
    args = get_arguments()
    env = OvercookedGymEnv(p1=DummyAgent(), args=args) #make('overcooked_ai.agents:OvercookedGymEnv-v0', layout='asymmetric_advantages', encoding_fn=encode_state, args=args)
    print(check_env(env))
    env.setup_visualization()
    env.reset()
    env.render()
    done = False
    while not done:
        obs, reward, done, info = env.step( Action.ACTION_TO_INDEX[np.random.choice(Action.ALL_ACTIONS)] )
        env.render()
