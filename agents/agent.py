from abc import ABC, abstractmethod
import torch as th
from typing import Tuple, Union


class OAIAgent(ABC):
    """
    A subset of stable baselines Base algorithm:
    https://stable-baselines3.readthedocs.io/en/master/modules/base.html#stable_baselines3.common.base_class.BaseAlgorithm
    For agents that aren't part of that library to play nicely with it
    """
    def __init__(self):
        super(OAIAgent, self).__init__()

    @abstractmethod
    def predict(self, obs: th.Tensor) -> Tuple[int, Union[th.Tensor, None]]:
        """
        Given an observation return the index of the action and the agent state if the agent is recurrent.
        Structure should be the same as agents created using stable baselines:
        https://stable-baselines3.readthedocs.io/en/master/modules/base.html#stable_baselines3.common.base_class.BaseAlgorithm.predict
        """
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass


class OAITrainer(ABC):
    """
    An abstract base class for trainer classes. Trainer classes should have two agents used in training.
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
        pass

    @abstractmethod
    def train_agents(self):
        pass

    @abstractmethod
    def save(self, path: Union[str, None]=None, tag: Union[str, None]=None):
        pass

    @abstractmethod
    def load(self, path: Union[str, None]=None, tag: Union[str, None]=None):
        pass