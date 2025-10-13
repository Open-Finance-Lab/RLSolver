from abc import ABC,abstractmethod
import gymnasium as gym
import torch

class _EnvBase(ABC):
    def __init__(self, num_envs: int):
        self.num_envs = num_envs

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def reward(self):
        pass

    @abstractmethod
    def step(self):
        pass

    def reset_parallel(self):
        return torch.vmap(lambda _: self.reset())()

    def reward_parallel(self):
        return torch.vmap(self.step)()

    def step_parallel(self):
        return torch.vmap(self.reward)()

