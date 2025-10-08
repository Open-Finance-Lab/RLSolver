import torch
from _env_base import EnvBase


class TSPEnv(EnvBase):
    """TSP environment with stateful interaction and parallel optimizations."""
    
    def __init__(self, nodes, device='cuda'):
        if nodes.dim() == 2:
            nodes = nodes.unsqueeze(0)
        
        batch_size = nodes.size(0)
        seq_len = nodes.size(1)
        
        super().__init__(
            num_envs=batch_size,
            state_dim=seq_len,
            action_dim=seq_len,
            if_discrete=True,
            device=device
        )
        
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.nodes = nodes
        
        self.encoded = None
        self.current_node = None
        self.first_node = None
        self.visited_mask = None
        self.tour = []
        self.step_count = 0
    
    def reward(self, nodes, tour):
        """Compute negative tour length for single instance.
        
        Args:
            nodes: [seq_len, 2]
            tour: [seq_len]
        Returns:
            negative tour length (scalar)
        """
        tour_nodes = nodes[tour]
        diffs = tour_nodes[1:] - tour_nodes[:-1]
        distances = torch.norm(diffs, dim=1)
        last_to_first = torch.norm(tour_nodes[-1] - tour_nodes[0])
        return -(distances.sum() + last_to_first)
    
    def reset_env(self):
        """Reset environment state for stateful interaction."""
        self.current_node = None
        self.first_node = None
        self.visited_mask = self.reset_parallel()
        self.tour = []
        self.step_count = 0
        self.encoded = None
        return self.get_observation()
    
    def step_env(self, action):
        """Execute action and update environment state.
        
        Args:
            action: [batch_size] node indices
        Returns:
            observation: dict
            done: bool
        """
        self.current_node = action
        
        if self.first_node is None:
            self.first_node = action.clone()
        
        self.visited_mask = self.step_parallel(self.visited_mask, action)
        self.tour.append(action)
        self.step_count += 1
        
        done = (self.step_count >= self.seq_len)
        
        return self.get_observation(), done
    
    def get_observation(self):
        """Construct observation dict for policy."""
        action_mask = ~self.visited_mask
        
        obs = {
            'nodes': self.nodes,
            'current_node': self.current_node,
            'first_node': self.first_node,
            'action_mask': action_mask
        }
        
        if self.encoded is not None:
            obs['encoded'] = self.encoded
            
        return obs
    
    def get_tour_length(self):
        """Compute tour lengths after complete rollout.
        
        Returns:
            tour_lengths: [batch_size]
        """
        tour_indices = torch.stack(self.tour, dim=1)
        return -self.reward_parallel(self.nodes, tour_indices)