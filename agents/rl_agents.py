import gym
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
import torch as th
import wandb

from arguments import get_arguments
from agent import OAIAgent, SB3Wrapper, OAITrainer
from networks import OAISinglePlayerFeatureExtractor, OAIDoublePlayerFeatureExtractor
from overcooked_gym_env import OvercookedGymEnv
from overcooked_subtask_gym_env import OvercookedSubtaskGymEnv
from state_encodings import ENCODING_SCHEMES

# LSTM TASK
class TwoSingleLSTMAgentTrainer(OAITrainer):
    def __init__(self, args):
        super(SingleLSTMAgentTrainer, self).__init__(args)
        pass

    def train_agents(self, epochs=1000, exp_name=None):
        pass
    def get_agent(self, idx):
        pass

    def save(self, path=None, tag=None):
        pass

    def load(self, path=None, tag=None):
        pass


class SB3SingleAgentWrapper(SB3Wrapper):
    ''' A wrapper for a stable baselines 3 agents that controls a single player '''
    def predict(self, obs):
        return self.agent.predict(obs)

    def get_distribution(self, obs: th.Tensor):
        return self.agent.get_distribution(obs)


class SB3DoubleAgentWrapper(SB3Wrapper):
    ''' A wrapper for a stable baselines agents that controls both players (i.e. outputs an action for both) '''
    def predict(self, obs):
        action, state = self.agent.predict(obs)
        return action[self.p_idx], state

    def get_distribution(self, obs: th.Tensor):
        return self.agent.get_distribution(obs)[self.p_idx]


class SingleAgentTrainer(OAITrainer):
    ''' Train an RL agent to play with a provided agent '''
    def __init__(self, teammate, teammate_idx, args, env=None):
        super(SingleAgentTrainer, self).__init__('single_agent', args)
        self.args = args
        self.device = args.device
        self.encoding_fn = ENCODING_SCHEMES[args.encoding_fn]
        self.t_idx, self.p_idx = teammate_idx, (teammate_idx + 1) % 2
        p_kwargs = {'p1': teammate} if teammate_idx == 0 else {'p2': teammate}
        kwargs = {'shape_rewards': True, 'obs_type': th.tensor, 'args': args}
        self.env = env or OvercookedGymEnv(**p_kwargs, **kwargs)

        policy_kwargs = dict(
            features_extractor_class=OAISinglePlayerFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )
        sb3_agent = PPO("MultiInputPolicy", self.env, policy_kwargs=policy_kwargs, verbose=1)
        learner = self.wrap_agent(sb3_agent)
        self.agents = [teammate, learner] if teammate_idx == 0 else [learner, teammate]

    def wrap_agent(self, sb3_agent):
        return SB3SingleAgentWrapper(sb3_agent, f'rl_single_agent_p{self.p_idx + 1}', self.p_idx, self.args)

    def train_agents(self, epochs=1000, exp_name=None):
        exp_name = exp_name or self.args.exp_name
        run = wandb.init(project="overcooked_ai_test", entity=self.args.wandb_ent, dir=str(self.args.base_dir / 'wandb'),
                         reinit=True, name=exp_name + '_' + self.agents[self.p_idx].name, mode=self.args.wandb_mode)
        for i in range(2):
            self.agents[i].policy.train()
        best_cum_rew = 0
        best_path, best_tag = None, None
        for epoch in range(epochs):
            if epoch % 10 == 0:
                cum_rew = self.env.evaluate(self.agents[self.p_idx], num_trials=1)
                print(f'Episode eval at epoch {epoch}: {cum_rew}')
                wandb.log({'eval_true_reward': cum_rew, 'epoch': epoch})
                if cum_rew > best_cum_rew:
                    best_path, best_tag = self.save()
                    best_cum_rew = cum_rew
            self.agents[self.p_idx].learn(total_timesteps=10000)
        if best_path is not None:
            self.load(best_path, best_tag)
        run.finish()

    def get_agent(self, idx):
        if idx != self.p_idx:
            raise ValueError(f'This trainer only trained a player {self.p_idx + 1} agent, '
                             f'and therefore cannot return a {self.t_idx + 1} agent')
        return self.agents[self.p_idx]


class TwoSingleAgentsTrainer(OAITrainer):
    ''' Train two independent RL agents to play with each other '''
    def __init__(self, args):
        super(TwoSingleAgentsTrainer, self).__init__('two_single_agents', args)
        self.device = args.device
        self.args = args
        self.encoding_fn = ENCODING_SCHEMES[args.encoding_fn]
        kwargs = {'shape_rewards': True, 'obs_type': th.tensor, 'args': args}
        self.envs = [OvercookedGymEnv(p2='temp', **kwargs), OvercookedGymEnv(p1='temp', **kwargs)]

        policy_kwargs = dict(
            features_extractor_class=OAISinglePlayerFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )
        agents = [PPO("MultiInputPolicy", self.envs[0], policy_kwargs=policy_kwargs, verbose=1),
                  PPO("MultiInputPolicy", self.envs[1], policy_kwargs=policy_kwargs, verbose=1)]
        self.agents = [SB3SingleAgentWrapper(agents[i], f'rl_two_single_agents_p{i + 1}', i, args) for i in range(2)]

        self.eval_env = OvercookedGymEnv(p1=self.agents[0], p2=self.agents[1], args=args)
        self.envs[0].set_agent(self.agents[1], idx=1)
        self.envs[1].set_agent(self.agents[0], idx=0)

    def train_agents(self, epochs=1000, exp_name=None):
        exp_name = exp_name or self.args.exp_name
        run = wandb.init(project="overcooked_ai_test", entity=self.args.wandb_ent, dir=str(self.args.base_dir / 'wandb'),
                         reinit=True, name=exp_name + '_rl_two_single_agents', mode=self.args.wandb_mode)
        for i in range(2):
            self.agents[i].policy.train()
        best_cum_rew = 0
        best_path, best_tag = None, None
        for epoch in range(epochs):
            self.agents[0].learn(total_timesteps=1000)
            self.agents[1].learn(total_timesteps=1000)
            if epoch % 10 == 0:
                cum_rew = self.eval_env.run_full_episode()
                print(f'Episode eval at epoch {epoch}: {cum_rew}')
                wandb.log({'eval_true_reward': cum_rew, 'epoch': epoch})
                if cum_rew > best_cum_rew:
                    best_path, best_tag = self.save()
                    best_cum_rew = cum_rew
        if best_path is not None:
            self.load(best_path, best_tag)
        run.finish()

    def get_agent(self, p_idx):
        return self.agents[p_idx]


class OneDoubleAgentTrainer(OAITrainer):
    ''' Train an RL agent to play both agents '''
    def __init__(self, args):
        super(OneDoubleAgentTrainer, self).__init__('one_double_agent', args)
        self.device = args.device
        self.args = args
        self.env = OvercookedGymEnv(obs_type= th.tensor, shape_rewards=True, args=args)
        policy_kwargs = dict(
            features_extractor_class=OAIDoublePlayerFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )
        self.agent = PPO("MultiInputPolicy", self.env, policy_kwargs=policy_kwargs, verbose=1)
        self.agents = [SB3DoubleAgentWrapper(self.agent, f'rl_one_double_agent_p{i + 1}', i, args) for i in range(2)]
        self.eval_env = OvercookedGymEnv(args=args)

    def train_agents(self, epochs=1000, exp_name=None):
        exp_name = exp_name or self.args.exp_name
        run = wandb.init(project="overcooked_ai_test", entity=self.args.wandb_ent, dir=str(self.args.base_dir / 'wandb'),
                         reinit=True, name=exp_name + '_rl_double_agent', mode=self.args.wandb_mode)
        self.agent.policy.train()
        best_cum_rew = 0
        best_path, best_tag = None, None
        for epoch in range(epochs):
            self.agent.learn(total_timesteps=10000)
            if epoch % 10 == 0:
                cum_rew = self.run_full_episode()
                print(f'Episode eval at epoch {epoch}: {cum_rew}')
                wandb.log({'eval_true_reward': cum_rew, 'epoch': epoch})
                if cum_rew > best_cum_rew:
                    best_path, best_tag = self.save()
                    best_cum_rew = cum_rew
        if best_path is not None:
            self.load(best_path, best_tag)
        run.finish()

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
        agent = SB3DoubleAgentWrapper(self.agent, idx)
        agent.set_name(f'rl_double_agent_p{idx + 1}')
        return agent



if __name__ == '__main__':
    args = get_arguments()
    oda = OneDoubleAgentTrainer(args)
    oda.train_agents(epochs=200)
    tsa = TwoSingleAgentsTrainer(args)
    tsa.train_agents(epochs=200)
    lstm = TwoSingleLSTMAgentTrainer(args)
    lstm.train_agents(epochs=200)


