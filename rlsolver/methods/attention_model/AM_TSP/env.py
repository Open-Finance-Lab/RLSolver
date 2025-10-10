import torch
from rlsolver.envs._env_base import EnvBase

class TSPEnv(EnvBase):
    
    def __init__(self, nodes, device='cuda'):
        
        if nodes.dim() == 2:
            nodes = nodes.unsqueeze(0)
        
        batch_size = nodes.size(0)
        seq_len = nodes.size(1)
        
        super().__init__(
            num_envs=batch_size,
            state_dim=seq_len * 2,
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
    
    @staticmethod
    def _compute_single_tour_length(nodes, tour_indices):
        """Compute tour length for a single instance.
        
        Args:
            nodes: [seq_len, 2]
            tour_indices: [seq_len]
        Returns:
            tour_length: scalar
        """
        tour_nodes = nodes[tour_indices]
        diffs = tour_nodes[1:] - tour_nodes[:-1]
        distances = torch.norm(diffs, dim=1)
        last_to_first = torch.norm(tour_nodes[-1] - tour_nodes[0])
        return distances.sum() + last_to_first
    
    def reset(self):
        """Reset environment to initial state."""
        self.current_node = None
        self.first_node = None
        self.visited_mask = torch.zeros(
            self.batch_size, self.seq_len,
            dtype=torch.bool, device=self.device
        )
        self.tour = []
        self.step_count = 0
        self.encoded = None
        return self.get_observation()
    
    def step(self, action):
        
        self.current_node = action
        if self.first_node is None:
            self.first_node = action.clone()
        
        self.visited_mask.scatter_(1, action.unsqueeze(1), True)
        self.tour.append(action)
        self.step_count += 1
        
        done = torch.full(
            (self.batch_size,),
            self.step_count >= self.seq_len,
            dtype=torch.bool,
            device=self.device
        )
        
    
        if done.all():
            reward_val = self.reward()
        else:
            reward_val = torch.zeros(self.batch_size, device=self.device)
        
        return self.get_observation(), reward_val, done
    
    def get_observation(self):
        """Get current observation."""
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
    
    def reward(self):
        
        tour_tensor = torch.stack(self.tour, dim=1)
        tour_lengths = torch.vmap(self._compute_single_tour_length)(
            self.nodes, tour_tensor
        )
        return -tour_lengths
    
    def reward_parallel(self, tours):
        
        batch_size, num_tours, seq_len = tours.shape
        
        nodes_expanded = self.nodes.unsqueeze(1).expand(
            batch_size, num_tours, seq_len, 2
        ).reshape(batch_size * num_tours, seq_len, 2)
        tours_flat = tours.reshape(batch_size * num_tours, seq_len)
        
        tour_lengths = torch.vmap(self._compute_single_tour_length)(
            nodes_expanded, tours_flat
        )
        
        tour_lengths = tour_lengths.view(batch_size, num_tours)
        return -tour_lengths
    
    def _setup_parallel_rollout(self, model, num_rollouts):
        """Setup parallel rollout for POMO or greedy evaluation."""
        embedded = model.network.embedding(self.nodes)
        encoded = model.network.encoder(embedded)
        
        encoded_expanded = encoded.unsqueeze(1).expand(
            self.batch_size, num_rollouts, self.seq_len, -1
        )
        nodes_expanded = self.nodes.unsqueeze(1).expand(
            self.batch_size, num_rollouts, self.seq_len, 2
        )
        
        encoded_flat = encoded_expanded.contiguous().view(
            self.batch_size * num_rollouts, self.seq_len, -1
        )
        nodes_flat = nodes_expanded.contiguous().view(
            self.batch_size * num_rollouts, self.seq_len, 2
        )
        
        visited_mask = torch.zeros(
            self.batch_size * num_rollouts, self.seq_len,
            dtype=torch.bool, device=self.device
        )
        
        flat_indices = torch.arange(
            self.batch_size * num_rollouts, device=self.device
        )
        pomo_indices = torch.arange(
            num_rollouts, device=self.device
        ).repeat(self.batch_size) % self.seq_len
        
        first_node = pomo_indices
        current_node = pomo_indices
        visited_mask[flat_indices, current_node] = True
        
        actions_tensor = torch.empty(
            self.batch_size * num_rollouts, self.seq_len,
            dtype=torch.long, device=self.device
        )
        actions_tensor[:, 0] = current_node
        
        return (encoded_flat, nodes_flat, visited_mask, flat_indices, 
                first_node, current_node, actions_tensor)
    
    def rollout_pomo(self, model, pomo_size=None):
        """Parallel POMO rollout using reward_parallel."""
        if pomo_size is None:
            pomo_size = self.seq_len
        
        (encoded_flat, nodes_flat, visited_mask, flat_indices, 
         first_node, current_node, actions_tensor) = self._setup_parallel_rollout(
            model, pomo_size
        )
        
        log_probs_tensor = torch.empty(
            self.batch_size * pomo_size, self.seq_len - 1,
            device=self.device
        )
        
        for step in range(1, self.seq_len):
            obs = {
                'nodes': nodes_flat,
                'current_node': current_node,
                'first_node': first_node,
                'action_mask': ~visited_mask,
                'encoded': encoded_flat
            }
            
            action, log_prob = model.get_action(obs, deterministic=False)
            current_node = action
            visited_mask[flat_indices, action] = True
            
            log_probs_tensor[:, step - 1] = log_prob
            actions_tensor[:, step] = action
        
        actions_reshaped = actions_tensor.view(
            self.batch_size, pomo_size, self.seq_len
        )
        rewards = self.reward_parallel(actions_reshaped)
        tour_lengths = -rewards
        
        log_probs = log_probs_tensor.view(
            self.batch_size, pomo_size, self.seq_len - 1
        )
        
        return tour_lengths, log_probs, actions_reshaped
    
    def rollout_greedy(self, model, num_rollouts=None):
        """Parallel greedy rollout using reward_parallel."""
        if num_rollouts is None:
            num_rollouts = self.seq_len
        
        (encoded_flat, nodes_flat, visited_mask, flat_indices, 
         first_node, current_node, actions_tensor) = self._setup_parallel_rollout(
            model, num_rollouts
        )
        
        for step in range(1, self.seq_len):
            obs = {
                'nodes': nodes_flat,
                'current_node': current_node,
                'first_node': first_node,
                'action_mask': ~visited_mask,
                'encoded': encoded_flat
            }
            
            logits = model.network(obs)
            action = logits.argmax(dim=-1)
            current_node = action
            visited_mask[flat_indices, action] = True
            actions_tensor[:, step] = action
        
        actions_reshaped = actions_tensor.view(
            self.batch_size, num_rollouts, self.seq_len
        )
        rewards = self.reward_parallel(actions_reshaped)
        tour_lengths = -rewards
        
        best_indices = tour_lengths.argmin(dim=1)
        best_lengths = tour_lengths.gather(
            1, best_indices.unsqueeze(1)
        ).squeeze(1)
        best_tours = actions_reshaped.gather(
            1, best_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.seq_len)
        ).squeeze(1)
        
        return best_tours, best_lengths
