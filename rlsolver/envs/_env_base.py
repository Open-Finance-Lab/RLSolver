from abc import ABC,abstractmethod
import gymnasium as gym


class _EnvBase(ABC, gym.Wrapper):
    def __init__(self, num_envs: int, state_dim: int, action_dim: int):
        self.num_envs = num_envs
        self.state_dim = state_dim
        self.action_dim = action_dim

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
        pass

    def reward_parallel(self):
        pass

    def step_parallel(self):
        pass

