import gym
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
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
VEC_ENV_CLS = SubprocVecEnv


class SingleAgentTrainer(OAITrainer):
    ''' Train an RL agent to play with a provided agent '''
    def __init__(self, teammates, args, env=None, eval_env=None, use_lstm=False, hidden_dim=256, seed=None):
        super(SingleAgentTrainer, self).__init__('single_agent', args, seed=seed)
        self.args = args
        self.device = args.device
        self.use_lstm = use_lstm
        self.encoding_fn = ENCODING_SCHEMES[args.encoding_fn]
        self.teammates = teammates
        self.n_tm = len(teammates)
        env_kwargs = {'shape_rewards': True, 'args': args}
        if env is None:
            self.env = make_vec_env(OvercookedGymEnv, n_envs=args.n_envs, env_kwargs=env_kwargs, vec_env_cls=VEC_ENV_CLS)
            self.eval_env = OvercookedGymEnv(shape_rewards=False, args=args)
        else:
            self.env = env
            self.eval_env = eval_env
        self.use_subtask_eval = (type(eval_env) == OvercookedSubtaskGymEnv)

        policy_kwargs = dict(
            features_extractor_class=OAISinglePlayerFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=hidden_dim),
            net_arch=[dict(pi=[hidden_dim, hidden_dim], vf=[hidden_dim, hidden_dim])]
        )
        if use_lstm:
            sb3_agent = RecurrentPPO('MultiInputLstmPolicy', self.env, policy_kwargs=policy_kwargs, verbose=1)
        else:
            sb3_agent = PPO('MultiInputPolicy', self.env, policy_kwargs=policy_kwargs, verbose=1)
        # sb3_agent.policy.to(self.device)
        self.learning_agent = self.wrap_agent(sb3_agent)
        self.agents = [self.learning_agent]
        # for agent in self.agents:
        #     agent.policy.to(self.device)

        # for i in range(self.args.n_envs):
        #     self.env.env_method('set_agent', self.teammates[np.random.randint(self.n_tm)], indices=i)

    def wrap_agent(self, sb3_agent):
        if self.use_lstm:
            agent = SB3LSTMWrapper(sb3_agent, f'rl_single_lstm_agent', self.args)
        else:
            agent = SB3Wrapper(sb3_agent, f'rl_single_agent', self.args)
        return agent

    def train_agents(self, total_timesteps=1e8, exp_name=None):
        exp_name = exp_name or self.args.exp_name
        run = wandb.init(project="overcooked_ai_test", entity=self.args.wandb_ent, dir=str(self.args.base_dir / 'wandb'),
                         reinit=True, name=exp_name + '_' + self.learning_agent.name, mode=self.args.wandb_mode)
        if self.use_subtask_eval:
            self.num_success = 0
        while self.learning_agent.agent.num_timesteps < total_timesteps:
            self.set_new_teammates()
            self.learning_agent.learn(total_timesteps=EPOCH_TIMESTEPS)
            eval_teammate = self.teammates[np.random.randint(len(self.teammates))]
            if self.use_subtask_eval:
                self.eval_env.set_teammate(eval_teammate)
                mean_reward, all_successes = self.eval_env.evaluate(self.learning_agent)
                self.num_success = (self.num_success + 1) if all_successes else 0
                if self.num_success >= 3:
                    break
            else:
                self.evaluate(self.learning_agent, eval_teammate)

        run.finish()


class MultipleAgentsTrainer(OAITrainer):
    ''' Train two independent RL agents to play with each other '''
    def __init__(self, args, num_agents=1, use_lstm=False, hidden_dim=256, fcp_ck_rate=None, seed=None):
        '''
        Train multiple agents with each other.
        :param num_agents: Number of agents to train. num_agents=1 mean self-play, num_agents > 1 is population play
        :param args: Experiment arguments. See arguments.py
        :param use_lstm: Whether agents should use an lstm policy or not
        :param hidden_dim: hidden dimensions for agents
        :param fcp_ck_rate: If not none, rate to save agents. Used primarily to get agents for Fictitous Co-Play
        :param seed: Random see
        '''
        super(MultipleAgentsTrainer, self).__init__('two_single_agents', args, seed=seed)
        self.device = args.device
        self.args = args
        self.fcp_ck_rate = fcp_ck_rate

        env_kwargs = {'shape_rewards': True, 'args': args}
        self.env = make_vec_env(OvercookedGymEnv, n_envs=args.n_envs, env_kwargs={**env_kwargs})
        self.eval_env = OvercookedGymEnv(shape_rewards=Fasle, args=args)

        policy_kwargs = dict(
            features_extractor_class=OAISinglePlayerFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=hidden_dim),
            net_arch=[dict(pi=[hidden_dim, hidden_dim], vf=[hidden_dim, hidden_dim])]
        )

        self.agents = []
        if use_lstm:
            for i in range(num_agents):
                sb3_agent = RecurrentPPO('MultiInputLstmPolicy', self.env, policy_kwargs=policy_kwargs, verbose=1,
                                         n_steps=2048, batch_size=16)
                self.agents.append(SB3LSTMWrapper(sb3_agent, f'rl_multiagent_lstm_{i + 1}', args))
        else:
            for i in range(num_agents):
                sb3_agent = PPO("MultiInputPolicy", self.env, policy_kwargs=policy_kwargs, verbose=1)
                self.agents.append(SB3Wrapper(sb3_agent, f'rl_multiagent_{i + 1}', i, args))

        self.agents_in_training = np.ones(len(self.agents))
        self.agents_timesteps = np.zeros(len(self.agents))

        # for agent in self.agents:
        #     agent.policy.to(self.device)

    def train_agents(self, total_timesteps=1e8, exp_name=None):
        if self.fcp_ck_rate is not None:
            self.ck_list = []
            path, tag = self.save(tag=f'ck_{len(ck)}')
            self.ck_list.append((0, path, tag))
        exp_name = exp_name or self.args.exp_name
        run = wandb.init(project="overcooked_ai_test", entity=self.args.wandb_ent, dir=str(self.args.base_dir / 'wandb'),
                         reinit=True, name=exp_name + '_rl_two_single_agents', mode=self.args.wandb_mode)
        # Each agent should learn for `total_timesteps` steps. Keep training until all agents hit this threshold
        while any(self.agents_in_training):
            # Randomly select new teammates from population (can include learner)
            self.set_new_teammates()
            # Randomly choose agent that will learn this time
            learner_idx = np.random.choice(len(self.agents), p=self.agents_in_training)
            # Learn and update recoded timesteps for that agent
            self.agents[learner_idx].learn(total_timesteps=EPOCH_TIMESTEPS)
            self.agents_timesteps[learner_idx] += EPOCH_TIMESTEPS
            if self.agents_timesteps[learner_idx] > total_timesteps:
                self.agents_in_training[learner_idx] = 0
            # Evaluate
            eval_tm = self.teammates[np.random.randint(len(self.teammates))]
            mean_reward = self.evaluate(self.agents[learner_idx], eval_tm, timestep=np.sum(self.agents_timesteps))

            if self.fcp_ck_rate and len(self.agents) == 1:
                if self.agents_timesteps[0] // self.fcp_ck_rate > (len(self.ck_list) - 1):
                    path, tag = self.save(tag=f'ck_{len(ck)}')
                    self.ck_list.append((mean_reward, path, tag))
        run.finish()

    def get_fcp_agents(self):
        if len(self.ck_list) < 3:
            raise ValueError('Must have at least 3 checkpoints saved. Increase fcp_ck_rate or training length')
        agents = []
        best_score = 0
        best_path, best_tag = None, None
        for score, path, tag in self.ck_list:
            if score > best_score:
                best_score = score
                best_path, best_tag = path, tag
        best = self.load(best_path, best_tag)
        agents.append(best.get_agents())
        del best
        _, worst_path, worst_tag = self.ck_list[0]
        worst = self.load(worst_path, worst_tag)
        agents.append(worst.get_agents())
        del worst

        closest_to_mid_score = float('inf')
        mid_path, mid_tag = None, None
        for i, (score, path, tag) in enumerate(self.ck_list):
            if abs((best_score / 2) - score) < closest_to_mid_score:
                closest_to_mid_score = score
                mid_path, mid_tag = path, tag
        mid = self.load(mid_path, mid_tag)
        agents.append(mid.get_agents())
        del mid
        return agents


class Population:
    ''' A class to manage a population of agents '''
    def __init__(self, agents, args):
        self.agents = agents
        self.args = args

    def get_agents(self):
        return self.agents

    def save(self, path: str) -> None:
        args = get_args_to_save(self.args)
        agent_path = path + '_pop_agents_dir'
        Path(agent_path).mkdir(parents=True, exist_ok=True)

        save_dict = {'model_type': type(self.agent[0]), 'agent_paths': [], 'args': args}
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
        return cls(agents, args)

    @staticmethod
    def create_fcp_population(args):
        p1_agents = []
        p2_agents = []
        for use_lstm in [True, False]:
            # hidden_dim = 16
            seed = 8
            for hidden_dim in [16, 256]:
            #     for seed in [1, 20]:#, 300, 4000]:
                total_timesteps = 5e6
                ck_rate = max(1, int(total_timesteps / (25 * EPOCH_TIMESTEPS)))
                rl_sat = MultipleAgentsTrainer(args, use_lstm=use_lstm, hidden_dim=hidden_dim, fcp_ck_rate=ck_rate,
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
    # lstm = MultipleAgentsTrainer(args, use_lstm=True)
    # lstm.train_agents(total_timesteps=1e6)
    # tsa = MultipleAgentsTrainer(args)
    # tsa.train_agents(total_timesteps=1e6)
    # oda = DoubleAgentTrainer(args)
    # oda.train_agents(total_timesteps=1e6)
    p1, p2 = Population.create_fcp_population(args)
    p1 = Poulation.load(str(self.args.base_dir / 'agent_models' / 'population' / self.args.layout_name / 'p1s'))
    p2 = Poulation.load(str(self.args.base_dir / 'agent_models' / 'population' / self.args.layout_name / 'p2s'))



