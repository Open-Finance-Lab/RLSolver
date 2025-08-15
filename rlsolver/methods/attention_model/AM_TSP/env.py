# env.py
"""TSP Environment for non-autoregressive rollout with vmap optimizations."""

import torch
import torch.nn.functional as F
from functools import partial


class TSPEnv:
    """TSP environment for managing state and transitions."""
    
    def __init__(self, nodes, device='cuda'):
        """
        Args:
            nodes: FloatTensor [batch_size, seq_len, 2] or [seq_len, 2]
        """
        if nodes.dim() == 2:
            nodes = nodes.unsqueeze(0)
        
        self.batch_size = nodes.size(0)
        self.seq_len = nodes.size(1)
        self.device = device
        
        # Static data
        self.nodes = nodes  # [batch_size, seq_len, 2]
        
        # Cached encoder output (for efficiency during evaluation)
        self.encoded = None  # [batch_size, seq_len, embedding_size]
        
        # Dynamic state
        self.current_node = None  # [batch_size]
        self.first_node = None  # [batch_size]
        self.visited_mask = None  # [batch_size, seq_len]
        self.tour = []  # List of selected nodes
        self.step_count = 0
    
    def reset(self):
        """Reset environment to initial state."""
        self.current_node = None
        self.first_node = None
        self.visited_mask = torch.zeros(self.batch_size, self.seq_len,
                                       dtype=torch.bool, device=self.device)
        self.tour = []
        self.step_count = 0
        self.encoded = None  # Reset cached encoder output
        return self.get_observation()
    
    def get_observation(self):
        """Get current observation for the network.
        
        Returns:
            dict with:
                - nodes: all node coordinates [batch_size, seq_len, 2]
                - current_node: current node index [batch_size] or None
                - first_node: first node index [batch_size] or None
                - action_mask: available actions [batch_size, seq_len]
                - encoded: cached encoder output [batch_size, seq_len, embedding_size] or None
        """
        # Action mask: True for available (unvisited) nodes
        action_mask = ~self.visited_mask
        
        obs = {
            'nodes': self.nodes,
            'current_node': self.current_node,
            'first_node': self.first_node,
            'action_mask': action_mask
        }
        
        # Include cached encoder output if available
        if self.encoded is not None:
            obs['encoded'] = self.encoded
            
        return obs
    
    def step(self, action):
        """Take a step in the environment.
        
        Args:
            action: Selected node indices [batch_size]
            
        Returns:
            observation: Next observation
            done: Whether episode is finished
        """
        # Update state
        self.current_node = action
        
        if self.first_node is None:
            self.first_node = action.clone()
        
        # Mark as visited
        self.visited_mask.scatter_(1, action.unsqueeze(1), True)
        self.tour.append(action)
        self.step_count += 1
        
        # Check if done
        done = (self.step_count >= self.seq_len)
        
        return self.get_observation(), done
    
    def get_tour_length(self):
        """Calculate total tour length after completion using vmap.
        
        Returns:
            lengths: FloatTensor [batch_size]
        """
        if len(self.tour) != self.seq_len:
            raise ValueError("Tour not complete")
        
        # Stack tour indices
        tour_indices = torch.stack(self.tour, dim=1)  # [batch_size, seq_len]
        
        # Use vmap for efficient tour length calculation
        def compute_single_tour_length(nodes, indices):
            """Compute tour length for a single instance."""
            # Gather nodes in tour order
            tour_nodes = nodes[indices]  # [seq_len, 2]
            
            # Calculate distances between consecutive nodes
            diffs = tour_nodes[1:] - tour_nodes[:-1]
            distances = torch.norm(diffs, dim=1)
            
            # Add distance from last to first
            last_to_first = torch.norm(tour_nodes[-1] - tour_nodes[0])
            
            return distances.sum() + last_to_first
        
        # Apply vmap across batch dimension
        total_lengths = torch.vmap(compute_single_tour_length)(self.nodes, tour_indices)
        
        return total_lengths


class VectorizedTSPEnv:
    """Fully vectorized TSP environment for parallel rollouts."""
    
    def __init__(self, nodes, device='cuda'):
        """
        Args:
            nodes: FloatTensor [batch_size, seq_len, 2]
        """
        self.batch_size = nodes.size(0)
        self.seq_len = nodes.size(1)
        self.device = device
        self.nodes = nodes
        
    def compute_all_tours(self, all_actions):
        """Compute tour lengths for complete action sequences using vmap.
        
        Args:
            all_actions: [batch_size, seq_len] - complete tour indices
            
        Returns:
            tour_lengths: [batch_size]
        """
        def compute_tour_length(nodes, actions):
            """Compute length for single tour."""
            tour_nodes = nodes[actions]
            
            # Distances between consecutive nodes
            diffs = tour_nodes[1:] - tour_nodes[:-1]
            distances = torch.norm(diffs, dim=1)
            
            # Distance from last to first
            last_to_first = torch.norm(tour_nodes[-1] - tour_nodes[0])
            
            return distances.sum() + last_to_first
        
        # Vectorize across batch
        tour_lengths = torch.vmap(compute_tour_length)(self.nodes, all_actions)
        return tour_lengths
    
    def batch_step(self, visited_mask, actions):
        """Vectorized step function for multiple environments.
        
        Args:
            visited_mask: [batch_size, seq_len] - current visited states
            actions: [batch_size] - selected actions
            
        Returns:
            new_visited_mask: [batch_size, seq_len]
        """
        # Create new mask with vmap
        def update_mask(mask, action):
            new_mask = mask.clone()
            new_mask[action] = True
            return new_mask
        
        new_visited_mask = torch.vmap(update_mask)(visited_mask, actions)
        return new_visited_mask