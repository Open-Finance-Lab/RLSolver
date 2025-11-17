import torch
import networkx as nx
from tqdm import tqdm
from torch_geometric.utils import from_networkx
from ._env_base import _EnvBase
from typing import Tuple, Dict, Any, Optional
import numpy as np


class PIGNNMaxCutEnv(_EnvBase):
    """
    Environment for Max-Cut problem using PIGNN approach.
    Generates d-regular graphs and provides physics-inspired evaluation.
    """

    def __init__(self, num_envs: int = 1, num_graphs: int = 1000, num_nodes: int = 100,
                 node_degree: int = 3, in_dim: int = 1, seed: int = 0):
        super().__init__(num_envs)

        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self.node_degree = node_degree
        self.in_dim = in_dim
        self.seed = seed

        # Generate graph pool
        self.graph_pool = self._generate_graph_pool()
        self.current_graph_idx = 0
        self.current_state = None

        # Physics parameters
        self.P = 0.7632  # Approximation constant for Max-Cut

    def _generate_graph_pool(self):
        """Generate pool of d-regular graphs."""
        graph_pool = []

        for i in tqdm(range(self.num_graphs), desc=f'Generating {self.num_graphs} d-regular graphs...'):
            g = nx.random_regular_graph(d=self.node_degree, n=self.num_nodes, seed=self.seed + i)
            pyg = from_networkx(g)
            pyg.x = torch.randn(self.num_nodes, self.in_dim)
            graph_pool.append(pyg)

        return graph_pool

    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset environment with a new graph."""
        graph = self.graph_pool[self.current_graph_idx % len(self.graph_pool)]
        self.current_graph_idx += 1

        # Random initial binary state
        self.current_state = torch.randint(0, 2, (self.num_nodes,), dtype=torch.float32)

        return {
            'edge_index': graph.edge_index,
            'node_features': graph.x,
            'current_state': self.current_state,
            'num_nodes': self.num_nodes
        }

    def step(self, action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict[str, Any]]:
        """
        Perform one step in the environment.

        Args:
            action: Binary tensor of shape [num_nodes] representing node assignments

        Returns:
            observation, reward, done, info
        """
        self.current_state = action.float()

        # Calculate physics-inspired reward (negative Hamiltonian)
        reward = self._calculate_maxcut_reward(self.graph_pool[
            (self.current_graph_idx - 1) % len(self.graph_pool)].edge_index,
            self.current_state)

        done = True  # Each step is a complete solution for Max-Cut

        obs = {
            'edge_index': self.graph_pool[(self.current_graph_idx - 1) % len(self.graph_pool)].edge_index,
            'node_features': self.graph_pool[(self.current_graph_idx - 1) % len(self.graph_pool)].x,
            'current_state': self.current_state,
            'num_nodes': self.num_nodes
        }

        info = {
            'approximation_ratio': self._calculate_approximation_ratio(reward),
            'cut_size': -reward
        }

        return obs, reward, done, info

    def reward(self, state: torch.Tensor, edge_index: torch.Tensor) -> float:
        """Calculate reward for given state."""
        return self._calculate_maxcut_reward(edge_index, state)

    def _calculate_maxcut_reward(self, edge_index: torch.Tensor, state: torch.Tensor) -> float:
        """
        Calculate physics-inspired reward using Hamiltonian.
        H = -Σ(2x_i*x_j - x_i - x_j) for edges (i,j)
        """
        i, j = edge_index
        hamiltonian = torch.sum(2 * state[i] * state[j] - state[i] - state[j])
        return -hamiltonian.item()  # Negative for maximization

    def _calculate_approximation_ratio(self, reward: float) -> float:
        """Calculate approximation ratio compared to theoretical upper bound."""
        cut_ub = (self.node_degree/4 + (self.P * np.sqrt(self.node_degree/4))) * self.num_nodes
        return abs(reward) / cut_ub


class PIGNNMISEnv(_EnvBase):
    """
    Environment for Maximum Independent Set problem using PIGNN approach.
    Generates d-regular graphs and provides physics-inspired evaluation.
    """

    def __init__(self, num_envs: int = 1, num_graphs: int = 1000, num_nodes: int = 100,
                 node_degree: int = 3, in_dim: int = 1, seed: int = 0, penalty_coeff: float = 2.0):
        super().__init__(num_envs)

        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self.node_degree = node_degree
        self.in_dim = in_dim
        self.seed = seed
        self.penalty_coeff = penalty_coeff

        # Generate graph pool
        self.graph_pool = self._generate_graph_pool()
        self.current_graph_idx = 0
        self.current_state = None

    def _generate_graph_pool(self):
        """Generate pool of d-regular graphs."""
        graph_pool = []

        for i in tqdm(range(self.num_graphs), desc=f'Generating {self.num_graphs} d-regular graphs...'):
            g = nx.random_regular_graph(d=self.node_degree, n=self.num_nodes, seed=self.seed + i)
            pyg = from_networkx(g)
            pyg.x = torch.randn(self.num_nodes, self.in_dim)
            graph_pool.append(pyg)

        return graph_pool

    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset environment with a new graph."""
        graph = self.graph_pool[self.current_graph_idx % len(self.graph_pool)]
        self.current_graph_idx += 1

        # Random initial binary state
        self.current_state = torch.randint(0, 2, (self.num_nodes,), dtype=torch.float32)

        return {
            'edge_index': graph.edge_index,
            'node_features': graph.x,
            'current_state': self.current_state,
            'num_nodes': self.num_nodes
        }

    def step(self, action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict[str, Any]]:
        """
        Perform one step in the environment.

        Args:
            action: Binary tensor of shape [num_nodes] representing node selection

        Returns:
            observation, reward, done, info
        """
        self.current_state = action.float()

        # Calculate physics-inspired reward
        reward = self._calculate_mis_reward(self.graph_pool[
            (self.current_graph_idx - 1) % len(self.graph_pool)].edge_index,
            self.current_state)

        done = True  # Each step is a complete solution for MIS

        obs = {
            'edge_index': self.graph_pool[(self.current_graph_idx - 1) % len(self.graph_pool)].edge_index,
            'node_features': self.graph_pool[(self.current_graph_idx - 1) % len(self.graph_pool)].x,
            'current_state': self.current_state,
            'num_nodes': self.num_nodes
        }

        info = {
            'independence_ratio': self._calculate_independence_ratio(),
            'violations': self._count_violations()
        }

        return obs, reward, done, info

    def reward(self, state: torch.Tensor, edge_index: torch.Tensor) -> float:
        """Calculate reward for given state."""
        return self._calculate_mis_reward(edge_index, state)

    def _calculate_mis_reward(self, edge_index: torch.Tensor, state: torch.Tensor) -> float:
        """
        Calculate physics-inspired reward using Hamiltonian.
        H = -Σx_i + P*Σ(x_i*x_j) for edges (i,j)
        """
        i, j = edge_index
        count_term = -torch.sum(state)
        penalty_term = torch.sum(self.penalty_coeff * (state[i] * state[j]))
        hamiltonian = count_term + penalty_term
        return -hamiltonian.item()  # Negative for maximization

    def _calculate_independence_ratio(self) -> float:
        """Calculate independence ratio."""
        return torch.sum(self.current_state).item() / self.num_nodes

    def _count_violations(self) -> int:
        """Count number of edge violations in current state."""
        edge_index = self.graph_pool[(self.current_graph_idx - 1) % len(self.graph_pool)].edge_index
        i, j = edge_index
        violations = torch.sum((self.current_state[i] == 1) & (self.current_state[j] == 1)).item()
        return violations