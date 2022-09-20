from abc import ABC, abstractmethod
import argparse
from arguments import get_args_to_save, set_args_from_load
from pathlib import Path
from state_encodings import ENCODING_SCHEMES
import torch as th
import torch.nn as nn
from typing import Tuple, Union
import stable_baselines3.common.distributions as sb3_distributions
from stable_baselines3.common.evaluation import evaluate_policy

class OAIAgent(nn.Module, ABC):
    """
    A smaller version of stable baselines Base algorithm with some small changes for my new agents
    https://stable-baselines3.readthedocs.io/en/master/modules/base.html#stable_baselines3.common.base_class.BaseAlgorithm
    Ensures that all agents play nicely with the environment
    """
    def __init__(self, name, args):
        super(OAIAgent, self).__init__()
        self.name = name
        # Player index and Teammate index
        self.encoding_fn = ENCODING_SCHEMES[args.encoding_fn]
        self.args = args
        # Must define a policy. The policy must implement a get_distribution(obs) that returns the action distribution
        self.policy = None

    @abstractmethod
    def predict(self, obs: th.Tensor, sample: bool=True) -> Tuple[int, Union[th.Tensor, None]]:
        """
        Given an observation return the index of the action and the agent state if the agent is recurrent.
        Structure should be the same as agents created using stable baselines:
        https://stable-baselines3.readthedocs.io/en/master/modules/base.html#stable_baselines3.common.base_class.BaseAlgorithm.predict
        """

    @abstractmethod
    def get_distribution(self, obs: th.Tensor) -> Union[th.distributions.Distribution, sb3_distributions.Distribution]:
        """
        Given an observation return the index of the action and the agent state if the agent is recurrent.
        Structure should be the same as agents created using stable baselines:
        https://stable-baselines3.readthedocs.io/en/master/modules/base.html#stable_baselines3.common.base_class.BaseAlgorithm.predict
        """

    def _get_constructor_parameters(self):
        return dict(name=self.name, args=self.args)

    def save(self, path: str) -> None:
        """
        Save model to a given location.
        :param path:
        """
        args = get_args_to_save(self.args)
        th.save({'state_dict': self.state_dict(), 'const_params': self._get_constructor_parameters(), 'args': args}, path)

    @classmethod
    def load(cls, path: str, args: argparse.Namespace) -> 'OAIAgent':
        """
        Load model from path.
        :param path: path to save to
        :param device: Device on which the policy should be loaded.
        :return:
        """
        device = args.device
        saved_variables = th.load(path, map_location=device)
        set_args_from_load(saved_variables['args'], args)
        saved_variables['const_params']['args'] = args
        # Create agent object
        model = cls(**saved_variables['const_params'])  # pytype: disable=not-instantiable
        # Load weights
        model.load_state_dict(saved_variables['state_dict'])
        model.to(device)
        model.reset(None)
        return model

class SB3Wrapper(OAIAgent):
    def __init__(self, agent, name, args):
        super(SB3Wrapper, self).__init__(name, args)
        self.agent = agent
        self.policy = self.agent.policy
        self.num_timesteps = 0

    def predict(self, obs, sample=True):
        return self.agent.predict(obs, deterministic=(not sample))

    def get_distribution(self, obs: th.Tensor):
        return self.agent.get_distribution(obs)

    def learn(self, total_timesteps):
        self.agent.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
        self.num_timesteps = self.agent.num_timesteps

    def save(self, path: str) -> None:
        """
        Save model to a given location.
        :param path:
        """
        args = get_args_to_save(self.args)
        th.save({'sb3_model_type': type(self.agent), 'const_params': self._get_constructor_parameters(), 'args': args},
                str(path) + '_non_sb3_data')
        self.agent.save(path)

    @classmethod
    def load(cls, path: str, args: argparse.Namespace, **kwargs) -> 'SB3Wrapper':
        """
        Load model from path.
        :param path: path to save to
        :param device: Device on which the policy should be loaded.
        :return:
        """
        device = args.device
        saved_variables = th.load(str(path) + '_non_sb3_data')
        set_args_from_load(saved_variables['args'], args)
        saved_variables['const_params']['args'] = args
        # Create agent object
        agent = saved_variables['sb3_model_type'].load(path)
        # Create wrapper object
        model = cls(agent=agent, **saved_variables['const_params'], **kwargs)  # pytype: disable=not-instantiable
        model.to(device)
        return model

class SB3LSTMWrapper(SB3Wrapper):
    ''' A wrapper for a stable baselines 3 agents that uses an lstm and controls a single player '''
    def __init__(self, agent, name, args):
        super(SB3LSTMWrapper, self).__init__(agent, name, args)
        self.lstm_states = None

    def predict(self, obs, sample=True, episode_starts=None):
        # TODO pass in episode starts
        episode_starts = episode_starts or np.ones((1,), dtype=bool)
        action, self.lstm_states = self.agent.predict(obs, state=self.lstm_states, episode_start=episode_starts,
                                                      deterministic=(not sample))
        return action, self.lstm_states

    def get_distribution(self, obs: th.Tensor):
        return self.agent.get_distribution(obs)


class OAITrainer(ABC):
    """
    An abstract base class for trainer classes.
    Trainer classes must have two agents that they can train using some paradigm
    """
    def __init__(self, name, args, seed=None):
        super(OAITrainer, self).__init__()
        self.name = name
        self.args = args
        self.ck_list = []
        if seed is not None:
            th.manual_seed(seed)

    def evaluate(self, eval_agent, eval_teammate, num_episodes=10, visualize=False, timestep=None):
        if visualize and not self.eval_env.visualization_enabled:
            self.eval_env.setup_visualization()
        self.eval_env.set_teammate(eval_teammate)
        mean_reward, std_reward = evaluate_policy(eval_agent, self.eval_env, n_eval_episodes=num_episodes, warn=False)
        timestep = timestep or eval_agent.num_timesteps
        print(f'Eval at timestep {timestep}: {mean_reward}')
        wandb.log({'eval_mean_reward': mean_reward, 'timestep': timestep})
        return mean_reward

    def set_new_teammates(self):
        for i in range(self.args.n_envs):
            self.env.env_method('set_teammate', self.teammates[np.random.randint(len(self.teammates))], indices=i)

    def get_agents(self) -> OAIAgent:
        """
        Structure should be the same as agents created using stable baselines:
        https://stable-baselines3.readthedocs.io/en/master/modules/base.html#stable_baselines3.common.base_class.BaseAlgorithm.predict
        """
        return self.agents

    def save(self, path: Union[str, None] = None, tag: Union[str, None] = None):
        ''' Saves each agent that the trainer is training '''
        path = path or self.args.base_dir / 'agent_models' / self.name / self.args.layout_name
        Path(path).mkdir(parents=True, exist_ok=True)
        tag = tag or self.args.exp_name
        for i in range(len(self.agents)):
            self.agents[i].save(path / (tag + f'_agent{i + 1}'))
        return path, tag

    def load(self, path: Union[str, None]=None, tag: Union[str, None]=None):
        ''' Loads each agent that the trainer is training '''
        path = path or self.args.base_dir / 'agent_models' / self.name / self.args.layout_name
        tag = tag or self.args.exp_name
        for i in range(2):
            self.agents[i] = self.agents[i].load(path / (tag + f'_p{i + 1}'), self.args)
