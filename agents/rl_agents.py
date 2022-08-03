import gym

from stable_baselines3 import PPO

from arguments import get_arguments
from networks import OAIFeatureExtractor
from overcooked_gym_env import OvercookedGymEnv
from state_encodings import encode_state, OAI_RL_encode_state



class TwoSingleAgents:
    def __init__(self, args):
        self.args = args
        self.env1 = OvercookedGymEnv(p2_agent='placeholder', layout='asymmetric_advantages', encoding_fn=encode_state, shape_rewards=True, args=args)
        self.env2 = OvercookedGymEnv(p1_agent='placeholder', layout='asymmetric_advantages', encoding_fn=encode_state, shape_rewards=True, args=args)

        policy_kwargs = dict(
            features_extractor_class=OAIFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )
        self.p1 = PPO("MultiInputPolicy", self.env1, policy_kwargs=policy_kwargs, verbose=1)
        self.p2 = PPO("MultiInputPolicy", self.env2, policy_kwargs=policy_kwargs, verbose=1)

        self.eval_env = OvercookedGymEnv(p1_agent=self.p1, p2_agent=self.p2, layout='asymmetric_advantages',
                                         encoding_fn=encode_state, args=args)
        self.env1.set_p2_agent(self.p2)
        self.env2.set_p1_agent(self.p1)

    def train_agents(self, epochs=3000):
        for epoch in range(epochs):
            self.p1.learn(total_timesteps=1000)
            self.p2.learn(total_timesteps=1000)
            if epoch % 10 == 0:
                self.cum_rew = self.eval_env.run_full_episode()
                print(f'Episode eval at epoch {epoch}: {cum_rew}')




if __name__ == '__main__':
    args = get_arguments()
    tsa = TwoSingleAgents(args)
    tsa.train_agents()

