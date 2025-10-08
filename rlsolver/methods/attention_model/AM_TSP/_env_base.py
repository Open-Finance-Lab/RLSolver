import torch
from abc import ABC, abstractmethod


class EnvBase(ABC):
    def __init__(self, num_envs, state_dim, action_dim, if_discrete, device):
        self.num_envs = num_envs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.if_discrete = if_discrete
        self.device = device
        
    @abstractmethod
    def reset(self):
        return torch.zeros(self.state_dim, dtype=torch.bool, device=self.device)
    @abstractmethod
    def step(self, state, action):
        new_state = state.clone()
        new_state[action] = True
        return new_state
    
    @abstractmethod
    def reward(self, nodes, tour):
        pass
    
    @abstractmethod
    def reset_parallel(self):
        dummy = torch.zeros(self.num_envs, device=self.device)
        return torch.vmap(lambda _: self.reset())(dummy)
    
    @abstractmethod
    def step_parallel(self, states, actions):
        return torch.vmap(self.step)(states, actions)
    
    @abstractmethod
    def reward_parallel(self, all_nodes, tours):
        return torch.vmap(self.reward)(all_nodes, tours)