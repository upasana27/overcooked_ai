import gym
from stable_baselines3 import PPO

from arguments import get_arguments
from agent import OAIAgent, OAITrainer
from networks import OAISinglePlayerFeatureExtractor, OAIDoublePlayerFeatureExtractor
from overcooked_gym_env import OvercookedGymEnv
from state_encodings import encode_state, OAI_RL_encode_state



class TwoSingleAgentsTrainer(OAITrainer):
    def __init__(self, args):
        super(TwoSingleAgentsTrainer, self).__init__(args)
        kwargs = {'layout': 'asymmetric_advantages', 'encoding_fn': encode_state, 'shape_rewards': True, 'args': args}
        self.envs = [OvercookedGymEnv(p2_agent='temp', **kwargs), OvercookedGymEnv(p1_agent='temp', **kwargs)]

        policy_kwargs = dict(
            features_extractor_class=OAISinglePlayerFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )
        self.agents = [PPO("MultiInputPolicy", self.env[0], policy_kwargs=policy_kwargs, verbose=1),
                       PPO("MultiInputPolicy", self.env[1], policy_kwargs=policy_kwargs, verbose=1)]

        self.eval_env = OvercookedGymEnv(p1_agent=self.agents[0], p2_agent=self.agents[1],
                                         layout='asymmetric_advantages', encoding_fn=encode_state, args=args)
        self.env[0].set_agent(self.agents[1], idx=1)
        self.env[1].set_agent(self.agents[0], idx=0)

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
        path = path or self.args.base_dir / 'agent_models' / 'RL_single_agents'
        tag = tag or self.args.exp_name
        for i in range(2):
            self.agents[i].save(path / tag + f'_p{i + 1}')

    def load(self, path=None, tag=None):
        path = path or self.args.base_dir / 'agent_models' / 'RL_single_agents'
        tag = tag or self.args.exp_name
        for i in range(2):
            self.agents[i].load(path / tag + f'_p{i + 1}')


class DoubleAgentWrapperForSinglePlayerMode(OAIAgent):
    def __init__(self, double_agent, idx, args):
        """
        double_agent must be a stable baselines stable_baselines3.common.base_class.BaseAlgorithm that outputs
        an action for each agent
        """
        super(DoubleAgentWrapperForSinglePlayerMode, self).__init__(args)
        self.double_agent = double_agent
        self.idx = idx

    def predict(self, obs):
        action, state = self.double_agent.predict(obs)
        return action[self.idx], state

    def save(self, path):
        print('In the future, please save the double agent trainer directly instead of this wrapper')
        self.double_agent.save()

    def load(self, path):
        print('In the future, please load the double agent trainer directly instead of this wrapper')
        self.double_agent.load()

class OneDoubleAgentTrainer(OAITrainer):
    def __init__(self, args):
        super(OneDoubleAgentTrainer, self).__init__(args)
        self.env = OvercookedGymEnv(layout='asymmetric_advantages', encoding_fn=encode_state, shape_rewards=True, args=args)

        policy_kwargs = dict(
            features_extractor_class=OAIDoublePlayerFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )
        self.agent = PPO("MultiInputPolicy", self.env, policy_kwargs=policy_kwargs, verbose=1)
        self.eval_env = OvercookedGymEnv(layout='asymmetric_advantages', encoding_fn=encode_state, args=args)

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
        return DoubleAgentWrapperForSinglePlayerMode(self, idx, args)

    def save(self, path=None, tag=None):
        path = path or self.args.base_dir / 'agent_models' / 'RL_double_agent'
        tag = tag or self.args.exp_name
        self.agent.save(path / tag)

    def load(self, path=None, tag=None):
        path = path or self.args.base_dir / 'agent_models' / 'RL_double_agent'
        tag = tag or self.args.exp_name
        self.agent.load(path / tag)



if __name__ == '__main__':
    args = get_arguments()
    oda = OneDoubleAgentTrainer(args)
    oda.train_agents(epochs=100)
    tsa = TwoSingleAgentsTrainer(args)
    tsa.train_agents(epochs=100)

