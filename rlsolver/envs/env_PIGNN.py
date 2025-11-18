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


class PIGNNGraphColoringEnv(_EnvBase):
    """
    Environment for Graph Coloring problem using PIGNN approach.
    Generates various types of graphs suitable for coloring and provides physics-inspired evaluation.
    """

    def __init__(self, num_envs: int = 1, num_graphs: int = 100, num_nodes: int = 25,
                 num_colors: int = 6, in_dim: int = 16, seed: int = 42,
                 lambda_entropy: float = 0.1, lambda_balance: float = 0.05):
        super().__init__(num_envs)

        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self.num_colors = num_colors
        self.in_dim = in_dim
        self.seed = seed
        self.lambda_entropy = lambda_entropy
        self.lambda_balance = lambda_balance

        # Generate diverse graph pool
        self.graph_pool = self._generate_graph_pool()
        self.current_graph_idx = 0
        self.current_state = None

    def _generate_graph_pool(self):
        """Generate diverse pool of graphs suitable for coloring."""
        graph_pool = []

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        print(f"Creating {self.num_graphs} graphs for graph coloring...")

        for i in tqdm(range(self.num_graphs)):
            # Generate different types of graphs for diversity
            if i % 3 == 0:
                # Erdős-Rényi random graph
                g = nx.erdos_renyi_graph(self.num_nodes, 0.25)
            elif i % 3 == 1:
                # Watts-Strogatz small-world graph
                g = nx.watts_strogatz_graph(self.num_nodes, 4, 0.3)
            else:
                # Barabási-Albert scale-free graph
                g = nx.barabasi_albert_graph(self.num_nodes, 3)

            # Ensure graph is connected
            if not nx.is_connected(g):
                components = list(nx.connected_components(g))
                for j in range(1, len(components)):
                    u = list(components[0])[0]
                    v = list(components[j])[0]
                    g.add_edge(u, v)

            # Generate symmetry-breaking features
            features = self._generate_symmetry_breaking_features(g)

            # Convert to PyTorch Geometric format
            pyg = from_networkx(g)
            pyg.x = features
            graph_pool.append(pyg)

        return graph_pool

    def _generate_symmetry_breaking_features(self, graph):
        """
        Generate symmetry-breaking input features for graph coloring.
        This matches the proven implementation.
        """
        # Random features
        random_features = torch.rand(self.num_nodes, self.in_dim // 2)

        # Node ID features to break symmetry
        node_id_features = torch.arange(self.num_nodes, dtype=torch.float32).unsqueeze(1) / self.num_nodes

        # Degree features normalized by max degree
        degree_features = torch.tensor([len(list(graph.neighbors(j))) for j in range(self.num_nodes)],
                                    dtype=torch.float32).unsqueeze(1) / self.num_nodes

        # Additional random features to reach desired dimension
        remaining_dim = self.in_dim - random_features.shape[1] - node_id_features.shape[1] - degree_features.shape[1]
        if remaining_dim > 0:
            additional_features = torch.rand(self.num_nodes, remaining_dim)
        else:
            additional_features = torch.empty(self.num_nodes, 0)

        # Concatenate all features
        features = torch.cat([random_features, node_id_features, degree_features, additional_features], dim=1)

        return features

    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset environment with a new graph."""
        graph = self.graph_pool[self.current_graph_idx % len(self.graph_pool)]
        self.current_graph_idx += 1

        # Random initial color assignment (one-hot encoded)
        initial_colors = torch.randint(0, self.num_colors, (self.num_nodes,))
        self.current_state = torch.nn.functional.one_hot(initial_colors, num_classes=self.num_colors).float()

        return {
            'edge_index': graph.edge_index,
            'node_features': graph.x,
            'current_state': self.current_state,
            'num_nodes': self.num_nodes,
            'num_colors': self.num_colors
        }

    def step(self, action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict[str, Any]]:
        """
        Perform one step in the environment.

        Args:
            action: One-hot encoded tensor of shape [num_nodes, num_colors] representing color assignments

        Returns:
            observation, reward, done, info
        """
        self.current_state = action

        # Calculate physics-inspired reward using Potts model Hamiltonian
        reward = self._calculate_graph_coloring_reward(
            self.graph_pool[(self.current_graph_idx - 1) % len(self.graph_pool)].edge_index,
            self.current_state
        )

        done = True  # Each step is a complete solution for graph coloring

        obs = {
            'edge_index': self.graph_pool[(self.current_graph_idx - 1) % len(self.graph_pool)].edge_index,
            'node_features': self.graph_pool[(self.current_graph_idx - 1) % len(self.graph_pool)].x,
            'current_state': self.current_state,
            'num_nodes': self.num_nodes,
            'num_colors': self.num_colors
        }

        info = {
            'used_colors': self._count_used_colors(),
            'conflicts': self._count_conflicts(),
            'efficiency': self._calculate_color_efficiency()
        }

        return obs, reward, done, info

    def reward(self, state: torch.Tensor, edge_index: torch.Tensor) -> float:
        """Calculate reward for given state."""
        return self._calculate_graph_coloring_reward(edge_index, state)

    def _calculate_graph_coloring_reward(self, edge_index: torch.Tensor, state: torch.Tensor) -> float:
        """
        Calculate physics-inspired reward using Potts model Hamiltonian.
        H = Conflict + λ_entropy * Entropy + λ_balance * Balance
        where Conflict = Σ(c_i·c_j) for edges (i,j)
        """
        device = state.device
        num_nodes = state.size(0)

        # Conflict loss - Potts model Hamiltonian
        product = torch.mm(state, state.t())
        adj = torch.zeros(num_nodes, num_nodes, device=device)
        i, j = edge_index
        adj[i, j] = 1.0
        conflict_loss = torch.sum(product * adj) / 2.0

        # Entropy regularization
        entropy = -torch.sum(state * torch.log(state + 1e-8), dim=1).mean()

        # Color balance regularization
        color_usage = torch.sum(state, dim=0)
        balance_loss = torch.var(color_usage)

        # Total Hamiltonian (lower is better, so return negative for maximization)
        total_loss = conflict_loss + self.lambda_entropy * entropy + self.lambda_balance * balance_loss

        return -total_loss.item()

    def _count_used_colors(self) -> int:
        """Count number of unique colors used."""
        color_assignments = torch.argmax(self.current_state, dim=1)
        return len(torch.unique(color_assignments))

    def _count_conflicts(self) -> int:
        """Count number of coloring conflicts."""
        edge_index = self.graph_pool[(self.current_graph_idx - 1) % len(self.graph_pool)].edge_index
        i, j = edge_index
        color_assignments = torch.argmax(self.current_state, dim=1)
        conflicts = torch.sum(color_assignments[i] == color_assignments[j]).item()
        return conflicts

    def _calculate_color_efficiency(self) -> float:
        """Calculate color efficiency ratio (lower is better)."""
        return float(self._count_used_colors()) / float(self.num_colors)

    def temperature_sampling_decode(self, probs: torch.Tensor, edge_index: torch.Tensor,
                                 temperature: float = 1.2, trials: int = 10) -> torch.Tensor:
        """
        Temperature sampling decoding strategy for better color assignments.
        Matches the proven implementation.
        """
        device = probs.device
        num_nodes = probs.size(0)
        best_colors = None
        best_used_colors = float('inf')

        # Multiple sampling trials
        for trial in range(trials):
            # Temperature sampling
            temp_logits = torch.log(probs + 1e-8) / temperature
            temp_probs = torch.softmax(temp_logits, dim=1)
            sampled_colors = torch.multinomial(temp_probs, 1).squeeze()

            # Build adjacency list
            adj = {}
            i, j = edge_index
            for idx in range(len(i)):
                if i[idx].item() not in adj:
                    adj[i[idx].item()] = []
                if j[idx].item() not in adj:
                    adj[j[idx].item()] = []
                adj[i[idx].item()].append(j[idx].item())
                adj[j[idx].item()].append(i[idx].item())

            # Sort nodes by confidence
            node_probs = temp_probs[torch.arange(num_nodes), sampled_colors]
            node_order = torch.argsort(node_probs, descending=True).tolist()

            colors = torch.zeros(num_nodes, dtype=torch.long)

            # Greedy color assignment
            for node in node_order:
                neighbor_colors = set()
                if node in adj:
                    for neighbor in adj[node]:
                        neighbor_colors.add(colors[neighbor].item())

                suggested_color = sampled_colors[node].item()
                if suggested_color not in neighbor_colors:
                    colors[node] = suggested_color
                else:
                    for color in range(self.num_colors):
                        if color not in neighbor_colors:
                            colors[node] = color
                            break

            # Post-processing for zero conflicts
            colors = self._post_process_colors(colors, edge_index, adj)

            used_colors_count = colors.unique().numel()
            if used_colors_count < best_used_colors:
                best_used_colors = used_colors_count
                best_colors = colors.clone()

        return best_colors

    def _post_process_colors(self, colors: torch.Tensor, edge_index: torch.Tensor, adj: Dict) -> torch.Tensor:
        """Post-process colors to guarantee zero conflicts."""
        colors = colors.to(edge_index.device)
        max_attempts = 50
        current_max_colors = self.num_colors

        for attempt in range(max_attempts):
            i, j = edge_index
            conflicts = (colors[i] == colors[j]).nonzero(as_tuple=True)[0]

            if len(conflicts) == 0:
                break

            for conflict_idx in conflicts:
                u, v = i[conflict_idx], j[conflict_idx]
                node_to_fix = u if np.random.random() < 0.5 else v

                neighbor_colors = set()
                if node_to_fix.item() in adj:
                    for neighbor in adj[node_to_fix.item()]:
                        neighbor_colors.add(colors[neighbor].item())

                fixed = False
                for color in range(current_max_colors):
                    if color not in neighbor_colors:
                        colors[node_to_fix] = color
                        fixed = True
                        break

                if not fixed:
                    colors[node_to_fix] = current_max_colors
                    current_max_colors += 1

        return colors.cpu()