import gym
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import RecurrentPPO
import torch as th
import wandb

from arguments import get_arguments
from agent import OAIAgent, SB3Wrapper, OAITrainer
from networks import OAISinglePlayerFeatureExtractor, OAIDoublePlayerFeatureExtractor
from overcooked_gym_env import OvercookedGymEnv
from overcooked_subtask_gym_env import OvercookedSubtaskGymEnv
from state_encodings import ENCODING_SCHEMES

EPOCH_TIMESTEPS = 10000

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


class SB3SingleAgentLSTMWrapper(SB3Wrapper):
    ''' A wrapper for a stable baselines 3 agents that uses an lstm and controls a single player '''
    def __init__(self, agent, name, p_idx, args):
        super(SB3SingleAgentLSTMWrapper, self).__init__(agent, name, p_idx, args)
        self.episode_starts = np.ones((1,), dtype=bool)
        self.lstm_states = None

    def predict(self, obs):
        action, self.lstm_states = self.agent.predict(obs, state=self.lstm_states, episode_start=self.episode_starts,
                                                      deterministic=True)
        return action, self.lstm_states

    def get_distribution(self, obs: th.Tensor):
        return self.agent.get_distribution(obs)



class SingleAgentTrainer(OAITrainer):
    ''' Train an RL agent to play with a provided agent '''
    def __init__(self, teammates, teammate_idx, args, env=None, eval_env=None, use_lstm=False, hidden_dim=256, seed=None):
        super(SingleAgentTrainer, self).__init__('single_agent', args, seed=seed)
        self.args = args
        self.device = args.device
        self.use_lstm = use_lstm
        self.encoding_fn = ENCODING_SCHEMES[args.encoding_fn]
        self.teammates = teammates
        self.n_tm = len(teammates)
        teammate = teammates[0]
        self.t_idx, self.p_idx = teammate_idx, (teammate_idx + 1) % 2
        p_kwargs = {'p1': teammate} if teammate_idx == 0 else {'p2': teammate}
        env_kwargs = {'shape_rewards': True, 'args': args}
        if env is not None:
            self.env = make_vec_env(OvercookedGymEnv, n_envs=args.n_envs, env_kwargs={**p_kwargs, **env_kwargs})
            self.eval_env = eval_env
        else:
            self.env = env or OvercookedGymEnv(**p_kwargs, **env_kwargs)
            self.eval_env = env

        fec = OAISinglePlayerFeatureExtractor
        policy_kwargs = dict(
            features_extractor_class=fec,
            features_extractor_kwargs=dict(features_dim=hidden_dim),
            net_arch=[dict(pi=[hidden_dim, hidden_dim], vf=[hidden_dim, hidden_dim])]
        )
        if use_lstm:
            sb3_agent = RecurrentPPO('MultiInputLstmPolicy', self.env, policy_kwargs=policy_kwargs, verbose=1)
        else:
            sb3_agent = PPO('MultiInputPolicy', self.env, policy_kwargs=policy_kwargs, verbose=1)
        sb3_agent.policy.to(self.device)
        learner = self.wrap_agent(sb3_agent)
        self.agents = [teammate, learner] if teammate_idx == 0 else [learner, teammate]
        for agent in self.agents:
            agent.policy.to(self.device)

        for i in range(self.args.n_envs):
            self.env.env_method('set_agent', self.teammates[np.random.randint(self.n_tm)], self.t_idx, indices=i)

    def wrap_agent(self, sb3_agent):
        if self.use_lstm:
            agent = SB3SingleAgentLSTMWrapper(sb3_agent, f'rl_single_lstm_agent_p{self.p_idx + 1}', self.p_idx, self.args)
        else:
            agent = SB3SingleAgentWrapper(sb3_agent, f'rl_single_agent_p{self.p_idx + 1}', self.p_idx, self.args)
        return agent

    def train_agents(self, total_timesteps=1e8, exp_name=None):
        exp_name = exp_name or self.args.exp_name
        run = wandb.init(project="overcooked_ai_test", entity=self.args.wandb_ent, dir=str(self.args.base_dir / 'wandb'),
                         reinit=True, name=exp_name + '_' + self.agents[self.p_idx].name, mode=self.args.wandb_mode)
        for i in range(2):
            self.agents[i].policy.train()
        epoch = 0
        best_score, scores = 0, []
        best_path, best_tag = None, None
        while self.agents[self.p_idx].agent.num_timesteps < total_timesteps:
            if epoch % 10 == 0:
                score, done_training = self.eval_env.evaluate(self.agents[self.p_idx], num_trials=1)
                scores.append(score)
                print(f'Episode eval at epoch {epoch}: {score}')
                wandb.log({'eval_true_reward': score, 'epoch': epoch, 'timestep': self.agents[self.p_idx].agent.num_timesteps})
                if score > best_score:
                    best_path, best_tag = self.save()
                    best_score = score
                if done_training:# or len(scores) > 10 and np.mean(scores[-3:]) <= np.mean(scores[-6:-3]): # no improvement
                    break

            self.agents[self.p_idx].learn(total_timesteps=EPOCH_TIMESTEPS)
            for i in range(self.args.n_envs):
                self.env.env_method('set_agent', self.teammates[np.random.randint(self.n_tm)], self.t_idx, indices=i)
            epoch += 1

        if best_path is not None:
            self.load(best_path, best_tag)
        run.finish()


class TwoSingleAgentsTrainer(OAITrainer):
    ''' Train two independent RL agents to play with each other '''
    def __init__(self, args, use_lstm=False, hidden_dim=256, fcp_ck_rate=None, seed=None):
        super(TwoSingleAgentsTrainer, self).__init__('two_single_agents', args, seed=seed)
        self.device = args.device
        self.args = args
        self.encoding_fn = ENCODING_SCHEMES[args.encoding_fn]
        self.fcp_ck_rate = fcp_ck_rate

        env_kwargs = {'shape_rewards': True, 'args': args}
        self.envs = [make_vec_env(OvercookedGymEnv, n_envs=args.n_envs, env_kwargs={**env_kwargs, 'p2': 'temp'}),
                     make_vec_env(OvercookedGymEnv, n_envs=args.n_envs, env_kwargs={**env_kwargs, 'p1': 'temp'})]

        policy_kwargs = dict(
            features_extractor_class=OAISinglePlayerFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=hidden_dim),
            net_arch=[dict(pi=[hidden_dim, hidden_dim], vf=[hidden_dim, hidden_dim])]
        )

        if use_lstm:
            agents = [RecurrentPPO('MultiInputLstmPolicy', self.envs[0], policy_kwargs=policy_kwargs, verbose=1,
                                   n_steps=2048, batch_size=64),
                      RecurrentPPO('MultiInputLstmPolicy', self.envs[1], policy_kwargs=policy_kwargs, verbose=1,
                                   n_steps=2048, batch_size=64)]
            self.agents = [SB3SingleAgentLSTMWrapper(agents[i], f'rl_two_single_lstm_agents_p{i + 1}', i, args) for i in
                           range(2)]
        else:
            agents = [PPO("MultiInputPolicy", self.envs[0], policy_kwargs=policy_kwargs, verbose=1),
                      PPO("MultiInputPolicy", self.envs[1], policy_kwargs=policy_kwargs, verbose=1)]
            self.agents = [SB3SingleAgentWrapper(agents[i], f'rl_two_single_agents_p{i + 1}', i, args) for i in
                           range(2)]

        self.eval_env = OvercookedGymEnv(p1=self.agents[0], p2=self.agents[1], args=args)
        self.envs[0].env_method('set_agent', self.agents[1], 1)
        self.envs[1].env_method('set_agent', self.agents[0], 0)
        for agent in self.agents:
            agent.policy.to(self.device)

    def train_agents(self, total_timesteps=1e8, exp_name=None):
        if self.fcp_ck_rate is not None:
            self.ck_list = []
        exp_name = exp_name or self.args.exp_name
        run = wandb.init(project="overcooked_ai_test", entity=self.args.wandb_ent, dir=str(self.args.base_dir / 'wandb'),
                         reinit=True, name=exp_name + '_rl_two_single_agents', mode=self.args.wandb_mode)
        for i in range(2):
            self.agents[i].policy.train()
        epoch = 0
        best_score, scores = 0, []
        best_path, best_tag = None, None
        while self.agents[0].agent.num_timesteps < total_timesteps:
            if epoch % 10 == 0:
                score = self.eval_env.run_full_episode()
                scores.append(score)
                print(f'Episode eval at epoch {epoch}: {score}')
                wandb.log({'eval_true_reward': score, 'epoch': epoch, 'timestep': self.agents[0].agent.num_timesteps})
                if score > best_score:
                    best_path, best_tag = self.save()
                    best_score = score
                if self.fcp_ck_rate is not None and epoch % self.fcp_ck_rate == 0:
                    path, tag = self.save(tag=f'epoch_{epoch}')
                    self.ck_list.append( (score, path, tag) )
            self.agents[0].learn(total_timesteps=EPOCH_TIMESTEPS)
            self.agents[1].learn(total_timesteps=EPOCH_TIMESTEPS)
            epoch += 1

        if best_path is not None:
            self.load(best_path, best_tag)
        run.finish()

    def get_fcp_agents(self):
        if len(self.ck_list) < 3:
            raise ValueError('Must have at least 3 checkpoints saved. Increase fcp_ck_rate or training length')
        p1_agents = []
        p2_agents = []
        best_score = 0
        best_path, best_tag = None, None
        for score, path, tag in self.ck_list:
            if score > best_score:
                best_score = score
                best_path, best_tag = path, tag
        best = self.load(best_path, best_tag)
        p1_agents.append(best.get_agent(0))
        p2_agents.append(best.get_agent(1))
        del best
        _, worst_path, worst_tag = self.ck_list[0]
        worst = self.load(worst_path, worst_tag)
        p1_agents.append(worst.get_agent(0))
        p2_agents.append(worst.get_agent(1))
        del worst

        closest_to_mid_score = float('inf')
        mid_path, mid_tag = None, None
        for i, (score, path, tag) in enumerate(self.ck_list):
            if abs((best_score / 2) - score) < closest_to_mid_score:
                closest_to_mid_score = score
                mid_path, mid_tag = path, tag
        mid = self.load(mid_path, mid_tag)
        p1_agents.append(mid.get_agent(0))
        p2_agents.append(mid.get_agent(p_idx))
        del mid
        return p1_agents, p2_agents


class OneDoubleAgentTrainer(OAITrainer):
    ''' Train an RL agent to play both agents '''
    def __init__(self, args, hidden_dim=256, seed=None):
        super(OneDoubleAgentTrainer, self).__init__('one_double_agent', args, seed=seed)
        self.device = args.device
        self.args = args

        env_kwargs = {'shape_rewards': True, 'args': args}
        self.env = make_vec_env(OvercookedGymEnv, n_envs=args.n_envs, env_kwargs=env_kwargs)

        policy_kwargs = dict(
            features_extractor_class=OAIDoublePlayerFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=hidden_dim),
            net_arch=[dict(pi=[hidden_dim, hidden_dim], vf=[hidden_dim, hidden_dim])]
        )
        self.base_agent = PPO("MultiInputPolicy", self.env, policy_kwargs=policy_kwargs, verbose=1)
        self.agents = [SB3DoubleAgentWrapper(self.base_agent, f'rl_one_double_agent_p{i + 1}', i, args) for i in range(2)]
        self.eval_env = OvercookedGymEnv(args=args)
        for agent in self.agents:
            agent.policy.to(self.device)

    def train_agents(self, total_timesteps=1e8, exp_name=None):
        exp_name = exp_name or self.args.exp_name
        run = wandb.init(project="overcooked_ai_test", entity=self.args.wandb_ent, dir=str(self.args.base_dir / 'wandb'),
                         reinit=True, name=exp_name + '_rl_double_agent', mode=self.args.wandb_mode)
        self.agents[0].policy.train()
        epoch = 0
        best_score, scores = 0, []
        best_path, best_tag = None, None
        while self.agents[0].agent.num_timesteps < total_timesteps:
            if epoch % 10 == 0:
                score = self.run_full_episode()
                scores.append(score)
                print(f'Episode eval at epoch {epoch}: {score}')
                wandb.log({'eval_true_reward': score, 'epoch': epoch, 'timestep': self.agents[0].agent.num_timesteps})
                if score > best_score:
                    best_path, best_tag = self.save()
                    best_score = score
            self.agents[0].learn(total_timesteps=EPOCH_TIMESTEPS)
            epoch += 1

        if best_path is not None:
            self.load(best_path, best_tag)
        run.finish()

    def run_full_episode(self):
        self.eval_env.reset()
        done = False
        total_reward = 0
        while not done:
            joint_action, _ = self.base_agent.predict(self.eval_env.get_obs())
            obs, reward, done, info = self.eval_env.step(joint_action)
            total_reward += np.sum(info['sparse_r_by_agent'])
        return total_reward


class Population:
    ''' A class to manage a population of agents '''
    def __init__(self, agents, p_idx, args):
        self.agents = agents
        self.p_idx = p_idx
        self.args = args

    def get_agents(self):
        return self.agents

    def save(self, path: str) -> None:
        args = get_args_to_save(self.args)
        agent_path = path + '_pop_agents_dir'
        Path(agent_path).mkdir(parents=True, exist_ok=True)

        save_dict = {'model_type': type(self.agent[0]), 'p_idx': self.p_idx, 'agent_paths': [], 'args': args}
        for i, agent in enumerate(agents):
            agent_path_i = agent_path + f'/agent_{i}'
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
            agent = saved_variables['model_type'].load(agent_path)
            agent.to(device)
            agents.append(agent)
        return cls(agents, save_dict['p_idx'], args)

    @staticmethod
    def create_fcp_population(args):
        p1_agents = []
        p2_agents = []
        for use_lstm in [True, False]:
            # hidden_dim = 16
            # seed = 0
            for hidden_dim in [16, 256]:
            #     for seed in [1, 20]:#, 300, 4000]:
                total_timesteps = 5e6
                ck_rate = max(1, int(total_timesteps / (25 * EPOCH_TIMESTEPS)))
                rl_sat = TwoSingleAgentsTrainer(args, use_lstm=use_lstm, hidden_dim=hidden_dim, fcp_ck_rate=ck_rate,
                                                seed=seed)
                rl_sat.train_agents(total_timesteps=total_timesteps)
                p1s, p2s = rl_sat.get_fcp_agents()
                p1_agents.extend(p1s)
                p2_agents.extend(p2s)
        pop_p1, pop_p2 = Population(p1_agents, 0, args), Population(p2_agents, 1, args)
        pop_p1.save(str(self.args.base_dir / 'agent_models' / 'population' / self.args.layout_name / 'p1s'))
        pop_p2.save(str(self.args.base_dir / 'agent_models' / 'population' / self.args.layout_name / 'p2s'))
        return pop_p1, pop_p2


if __name__ == '__main__':
    args = get_arguments()
    # lstm = TwoSingleAgentsTrainer(args, use_lstm=True)
    # lstm.train_agents(total_timesteps=1e6)
    # tsa = TwoSingleAgentsTrainer(args)
    # tsa.train_agents(total_timesteps=1e6)
    # oda = OneDoubleAgentTrainer(args)
    # oda.train_agents(total_timesteps=1e6)
    p1, p2 = Population.create_fcp_population(args)
    p1 = Poulation.load(str(self.args.base_dir / 'agent_models' / 'population' / self.args.layout_name / 'p1s'))
    p2 = Poulation.load(str(self.args.base_dir / 'agent_models' / 'population' / self.args.layout_name / 'p2s'))



