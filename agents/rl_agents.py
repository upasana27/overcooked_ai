import gym
from stable_baselines3 import PPO

from arguments import get_arguments
from agent import OAIAgent, OAITrainer
from networks import OAISinglePlayerFeatureExtractor, OAIDoublePlayerFeatureExtractor
from overcooked_gym_env import OvercookedGymEnv
from state_encodings import ENCODING_SCHEMES


class SingleAgentWrapper(OAIAgent):
    def __init__(self, agent, idx):
        """
        double_agent must be a stable baselines stable_baselines3.common.base_class.BaseAlgorithm that outputs
        an action for each agent
        """
        super(SingleAgentWrapper, self).__init__()
        self.agent = agent
        self.set_player_idx(idx)

    def predict(self, obs):
        return self.agent.predict(obs)

    def get_distribution(self, obs: th.Tensor):
        return self.agent.get_distribution(obs)

    def save(self, path):
        print('In the future, please save the agent using TwoSingleAgentsTrainer.save')
        self.agent.save(path + f'_p{self.player_idx + 1}')

    def load(self, path):
        print('In the future, please load the agent using TwoSingleAgentsTrainer.load')
        self.agent.load(path + f'_p{self.player_idx + 1}')

class TwoSingleAgentsTrainer(OAITrainer):
    def __init__(self, args):
        super(TwoSingleAgentsTrainer, self).__init__(args)
        self.args = args
        self.encoding_fn = ENCODING_SCHEMES[args.encoding_fn]
        kwargs = {'layout': args.layout_name, 'encoding_fn': self.encoding_fn, 'shape_rewards': True, 'args': args}
        self.envs = [OvercookedGymEnv(p2_agent='temp', **kwargs), OvercookedGymEnv(p1_agent='temp', **kwargs)]

        policy_kwargs = dict(
            features_extractor_class=OAISinglePlayerFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )
        self.agents = [PPO("MultiInputPolicy", self.envs[0], policy_kwargs=policy_kwargs, verbose=1),
                       PPO("MultiInputPolicy", self.envs[1], policy_kwargs=policy_kwargs, verbose=1)]

        self.eval_env = OvercookedGymEnv(p1_agent=self.agents[0], p2_agent=self.agents[1],
                                         layout=args.layout_name, encoding_fn=self.encoding_fn, args=args)
        self.envs[0].set_agent(self.agents[1], idx=1)
        self.envs[1].set_agent(self.agents[0], idx=0)

    def train_agents(self, epochs=3000):
        best_cum_rew = 0
        for epoch in range(epochs):
            self.agents[0].learn(total_timesteps=1000)
            self.agents[1].learn(total_timesteps=1000)
            if epoch % 10 == 0:
                cum_rew = self.eval_env.run_full_episode()
                print(f'Episode eval at epoch {epoch}: {cum_rew}')
                if cum_rew > best_cum_rew:
                    self.save()
                    best_cum_rew = cum_rew

    def get_agent(self, idx):
        return self.agents[idx]

    def save(self, path=None, tag=None):
        path = path or self.args.base_dir / 'agent_models' / 'RL_two_single_agents' / self.args.layout_name
        tag = tag or self.args.exp_name
        for i in range(2):
            self.agents[i].save(path / tag + f'_p{i + 1}')

    def load(self, path=None, tag=None):
        path = path or self.args.base_dir / 'agent_models' / 'RL_two_single_agents' / self.args.layout_name
        tag = tag or self.args.exp_name
        for i in range(2):
            self.agents[i].load(path / tag + f'_p{i + 1}')


class SingleAgentTrainer(OAITrainer):
    def __init__(self, teammate, teammate_idx, args):
        super(SingleAgentTrainer, self).__init__(args)
        self.args = args
        self.encoding_fn = ENCODING_SCHEMES[args.encoding_fn]
        self.teammate_idx = teammate_idx
        self.player_idx = (teammate_idx + 1) % 2
        kwargs = {'layout': args.layout_name, 'encoding_fn': self.encoding_fn, 'shape_rewards': True, 'args': args}
        self.env = OvercookedGymEnv(p1_agent=teammate, **kwargs) if teammate_idx == 0 else \
                   OvercookedGymEnv(p2_agent=teammate, **kwargs)

        policy_kwargs = dict(
            features_extractor_class=OAISinglePlayerFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )
        self.agents = [teammate, PPO("MultiInputPolicy", self.env, policy_kwargs=policy_kwargs, verbose=1)] \
                      if teammate_idx == 0 else \
                      [PPO("MultiInputPolicy", self.env, policy_kwargs=policy_kwargs, verbose=1), teammate]

        self.eval_env = OvercookedGymEnv(p1_agent=self.agents[0], p2_agent=self.agents[1],
                                         layout=args.layout_name, encoding_fn=self.encoding_fn, args=args)
    def train_agents(self, epochs=3000):
        # TODO after training set agents to best saved agent
        best_cum_rew = 0
        for epoch in range(epochs):
            self.agents[self.player_idx].learn(total_timesteps=1000)
            if epoch % 10 == 0:
                cum_rew = self.eval_env.run_full_episode()
                print(f'Episode eval at epoch {epoch}: {cum_rew}')
                if cum_rew > best_cum_rew:
                    self.save()
                    best_cum_rew = cum_rew

    def get_agent(self, idx):
        if idx != self.player_idx:
            raise ValueError(f'This trainer only trained a player {self.player_idx + 1} agent, '
                             f'and therefore cannot return a {self.teammate_idx + 1} agent')
        return self.agents[self.player_idx]

    def save(self, path=None, tag=None):
        path = path or self.args.base_dir / 'agent_models' / 'RL_single_agent' / self.args.layout_name
        tag = tag or self.args.exp_name
        self.agents[self.player_idx].save(path / tag + f'_p{self.player_idx + 1}')

    def load(self, path=None, tag=None):
        path = path or self.args.base_dir / 'agent_models' / 'RL_single_agent' / self.args.layout_name
        tag = tag or self.args.exp_name
        for i in range(2):
            self.agents[i].load(path / tag + f'_p{i + 1}')


class DoubleAgentWrapper(OAIAgent):
    def __init__(self, double_agent, idx):
        """
        double_agent must be a stable baselines stable_baselines3.common.base_class.BaseAlgorithm that outputs
        an action for each agent
        """
        super(DoubleAgentWrapper, self).__init__()
        self.double_agent = double_agent
        self.set_player_idx(idx)

    def predict(self, obs):
        action, state = self.double_agent.predict(obs)
        return action[self.idx], state

    def get_distribution(self, obs: th.Tensor):
        return self.double_agent.get_distribution(obs)[self.idx]

    def save(self, path):
        print('In the future, please save the agent using OneDoubleAgentTrainer.save')
        self.double_agent.save(path + f'_p{self.player_idx + 1}')

    def load(self, path):
        print('In the future, please load the agent using OneDoubleAgentTrainer.load')
        self.double_agent.load(path + f'_p{self.player_idx + 1}')

class OneDoubleAgentTrainer(OAITrainer):
    def __init__(self, args):
        super(OneDoubleAgentTrainer, self).__init__(args)
        self.args = args
        self.encoding_fn = ENCODING_SCHEMES[args.encoding_fn]
        self.env = OvercookedGymEnv(layout=args.layout_name, encoding_fn=self.encoding_fn, shape_rewards=True, args=args)

        policy_kwargs = dict(
            features_extractor_class=OAIDoublePlayerFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )
        self.agent = PPO("MultiInputPolicy", self.env, policy_kwargs=policy_kwargs, verbose=1)
        self.eval_env = OvercookedGymEnv(layout=args.layout_name, encoding_fn=self.encoding_fn, args=args)

    def train_agents(self, epochs=3000):
        best_cum_rew = 0
        for epoch in range(epochs):
            self.agent.learn(total_timesteps=1000)
            if epoch % 10 == 0:
                cum_rew = self.run_full_episode()
                print(f'Episode eval at epoch {epoch}: {cum_rew}')
                if cum_rew > best_cum_rew:
                    self.save()
                    best_cum_rew = cum_rew

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
        return DoubleAgentWrapper(self, idx)

    def save(self, path=None, tag=None):
        path = path or self.args.base_dir / 'agent_models' / 'RL_double_agent' / self.args.layout_name
        tag = tag or self.args.exp_name
        self.agent.save(path / tag)

    def load(self, path=None, tag=None):
        path = path or self.args.base_dir / 'agent_models' / 'RL_double_agent' / self.args.layout_name
        tag = tag or self.args.exp_name
        self.agent.load(path / tag)



if __name__ == '__main__':
    args = get_arguments()
    oda = OneDoubleAgentTrainer(args)
    oda.train_agents(epochs=100)
    tsa = TwoSingleAgentsTrainer(args)
    tsa.train_agents(epochs=100)

