from abc import ABC, abstractmethod
import torch as th
from typing import Tuple, Union
import stable_baselines3.common.distributions as sb3_distributions

class OAIAgent(ABC):
    """
    A smaller version of stable baselines Base algorithm with some small changes for my new agents
    https://stable-baselines3.readthedocs.io/en/master/modules/base.html#stable_baselines3.common.base_class.BaseAlgorithm
    Ensures that all agents play nicely with the environment
    """
    def __init__(self):
        super(OAIAgent, self).__init__()
        # Must define a policy. The policy must implement a get_distribution(obs) that returns the action distribution
        self.policy = None

    def set_player_idx(self, idx):
        self.player_idx = idx

    def step(self, state, joint_action):
        pass

    def reset(self, state, player_idx: int):
        self.set_player_idx(player_idx)

    @abstractmethod
    def predict(self, obs: th.Tensor) -> Tuple[int, Union[th.Tensor, None]]:
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

    @abstractmethod
    def save(self, path: str):
        """Save agent"""

    @abstractmethod
    def load(self, path: str):
        """Load agent"""


class OAITrainer(ABC):
    """
    An abstract base class for trainer classes.
    Trainer classes must have two agents that they can train using some paradigm
    """
    def __init__(self, args):
        super(OAITrainer, self).__init__()
        self.args = args

    @abstractmethod
    def get_agent(self, idx: int) -> OAIAgent:
        """
        Given an observation return the index of the action and the agent state if the agent is recurrent.
        Structure should be the same as agents created using stable baselines:
        https://stable-baselines3.readthedocs.io/en/master/modules/base.html#stable_baselines3.common.base_class.BaseAlgorithm.predict
        """

    @abstractmethod
    def train_agents(self):
        """Run the training regime on agents"""

    @abstractmethod
    def save(self, path: Union[str, None]=None, tag: Union[str, None]=None):
        """Saves each agent that the trainer is training"""

    @abstractmethod
    def load(self, path: Union[str, None]=None, tag: Union[str, None]=None):
        """Loads each agent that the trainer is training"""
