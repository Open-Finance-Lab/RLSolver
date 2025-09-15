from abc import ABC,abstractmethod
import gymnasium as gym


class _EnvBase(ABC, gym.Wrapper):
    def __init__(self, num_envs: int, state_dim: int, action_dim: int, if_discrete: bool):
        self.num_envs = num_envs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.if_discrete = if_discrete

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def reward(self):
        pass

    @abstractmethod
    def step(self, state, action):
        pass

    @abstractmethod
    def reset_parallel(self):
        pass

    @abstractmethod
    def reward_parallel(self):
        pass

    @abstractmethod
    def step_parallel(self, states, actions):
        pass

