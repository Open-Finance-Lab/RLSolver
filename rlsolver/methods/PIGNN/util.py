import math
import os
from typing import List, Optional

import torch
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data

from rlsolver.methods.util import calc_txt_files_with_prefixes

def hamiltonian_maxcut(edge_index, pred):
    i, j = edge_index
    hamiltonian = torch.sum(2 * pred[i] * pred[j] - pred[i] - pred[j])

    return hamiltonian

def eval_maxcut(edge_index, pred, d, n):
    # Return the value of the maximum cut and the approximation ratio (compared to the optimal solution)
    maxcut_energy = -hamiltonian_maxcut(edge_index, pred)

    # Calculate approximatio ratio
    P = 0.7632
    cut_ub = (d/4 + (P*math.sqrt(d/4))) * n
    approx_ratio = maxcut_energy / cut_ub 

    return maxcut_energy, approx_ratio

def hamiltonian_MIS(edge_index, pred, P=2):
    i, j = edge_index
    count_term = -pred.sum()
    penalty_term = torch.sum(P * (pred[i] * pred[j]))
    hamiltonian = count_term + penalty_term

    return hamiltonian

def eval_MIS(edge_index, pred, d, n):
    # Return the value of the maximum cut and the approximation ratio (compared to the optimal solution)
    mis_energy = -hamiltonian_MIS(edge_index, pred)

    # Check that the produced set is actually composed of independent nodes
    # get independent set
    ind_set = torch.where(pred == 1)[0]
    ind_set_nodes = torch.sort(ind_set)[0]

    # Check if there is an edge between any pair of nodes
    num_violations, problem_edges = 0, []
    for i in range(ind_set_nodes.size(0) - 1):
        for j in range(i + 1, ind_set_nodes.size(0)):
            edge = torch.tensor([ind_set_nodes[i], ind_set_nodes[j]], dtype=torch.long)
            if torch.any(torch.all(edge == edge_index.T, dim=1)):
                num_violations += 1
                problem_edges.append(edge)

    # Remove (greedily) the nodes from the MIS
    if len(problem_edges):
        problem_edges = torch.vstack(problem_edges).T
        postpred = ind_set_nodes[~torch.isin(ind_set_nodes, problem_edges[0].unique())]
    else:
        postpred = pred

    # Calculate independence number
    alpha = len(postpred) / n

    return mis_energy, torch.tensor(alpha)

def hamiltonian_graph_coloring(edge_index, pred, lambda_entropy=0.1, lambda_balance=0.05):
    """
    Physics-inspired Hamiltonian for graph coloring using Potts model.
    This function implements the conflict loss from the user's proven Graph Coloring implementation.

    Args:
        edge_index: Tensor of shape [2, num_edges] representing graph connectivity
        pred: Tensor of shape [num_nodes, num_colors] with softmax probabilities
        lambda_entropy: Entropy regularization coefficient
        lambda_balance: Color balance regularization coefficient

    Returns:
        hamiltonian: Total energy (lower is better)
    """
    device = pred.device
    num_nodes = pred.size(0)
    num_colors = pred.size(1)

    # Conflict loss - Potts model Hamiltonian
    # Penalize adjacent nodes having the same color
    product = torch.mm(pred, pred.t())
    adj = torch.zeros(num_nodes, num_nodes, device=device)
    i, j = edge_index
    adj[i, j] = 1.0
    conflict_loss = torch.sum(product * adj) / 2.0

    # Entropy regularization for diverse color assignment
    entropy_per_node = -torch.sum(pred * torch.log(pred + 1e-8), dim=1)
    entropy_loss = -torch.mean(entropy_per_node)

    # Color balance loss to encourage usage of all colors
    color_usage = torch.mean(pred, dim=0)
    target_usage = 1.0 / num_colors
    balance_loss = torch.nn.functional.mse_loss(color_usage, torch.full_like(color_usage, target_usage))

    # Total Hamiltonian (energy to minimize)
    hamiltonian = conflict_loss + lambda_entropy * entropy_loss + lambda_balance * balance_loss

    return hamiltonian

def eval_graph_coloring(edge_index, assignments, num_colors, num_nodes):
    """
    Evaluate graph coloring solution.

    Args:
        edge_index: Tensor of shape [2, num_edges] representing graph connectivity
        assignments: Tensor of shape [num_nodes, num_colors] with softmax probabilities
            or tensor of shape [num_nodes] with discrete color indices
        num_colors: Number of available colors
        num_nodes: Number of nodes in the graph

    Returns:
        coloring_energy: Negative of conflict-free score (higher is better)
        chromatic_ratio: Ratio of used colors to available colors
    """
    if assignments.dim() == 2:
        color_assignments = torch.argmax(assignments, dim=1)
    else:
        color_assignments = assignments.long()

    # Count violations (adjacent nodes with same color)
    i, j = edge_index
    violations = torch.sum(color_assignments[i] == color_assignments[j]).float()

    # Count unique colors used
    used_colors = color_assignments.unique().numel()

    # Energy is negative violations (we want to minimize violations)
    coloring_energy = -violations

    # Chromatic ratio: lower is better (closer to 1 means using all available colors efficiently)
    chromatic_ratio = used_colors / max(1, num_colors)

    return coloring_energy, torch.tensor(chromatic_ratio)


def _read_graph_from_txt(file_path: str) -> nx.Graph:
    """Load a graph from a Gset-style .txt file with 1-indexed nodes."""
    graph = nx.Graph()
    with open(file_path, "r") as handle:
        first_line = handle.readline()
        while first_line and first_line.strip().startswith("//"):
            first_line = handle.readline()
        if not first_line:
            raise ValueError(f"Empty graph file: {file_path}")
        parts = first_line.strip().split()
        if len(parts) < 2:
            raise ValueError(f"Invalid header in graph file: {file_path}")
        num_nodes = int(parts[0])
        graph.add_nodes_from(range(num_nodes))
        for line in handle:
            if not line.strip() or line.startswith("//"):
                continue
            items = line.strip().split()
            if len(items) < 2:
                continue
            node1, node2 = int(items[0]) - 1, int(items[1]) - 1
            weight = float(items[2]) if len(items) > 2 else 1.0
            graph.add_edge(node1, node2, weight=weight)
    return graph


def _create_graph_coloring_features(graph, feature_dim: int) -> torch.Tensor:
    """Create deterministic + random features matching training-time layout."""
    num_nodes = graph.number_of_nodes()
    if num_nodes == 0:
        raise ValueError("Graph must contain at least one node.")

    random_dim = max(1, feature_dim // 2)
    random_features = torch.rand(num_nodes, random_dim)

    node_indices = torch.arange(num_nodes, dtype=torch.float32).unsqueeze(1)
    norm = max(1.0, float(num_nodes - 1))
    node_id_features = node_indices / norm

    degree_values = torch.tensor(
        [graph.degree(i) for i in range(num_nodes)],
        dtype=torch.float32
    ).unsqueeze(1)
    degree_norm = max(1.0, degree_values.max().item())
    degree_features = degree_values / degree_norm

    remaining_dim = feature_dim - random_features.size(1) - node_id_features.size(1) - degree_features.size(1)
    if remaining_dim > 0:
        additional_features = torch.rand(num_nodes, remaining_dim)
        feature_blocks = [random_features, node_id_features, degree_features, additional_features]
    else:
        # Trim random features if feature_dim is very small
        random_features = random_features[:, :random_features.size(1) + remaining_dim]
        feature_blocks = [random_features, node_id_features, degree_features]

    features = torch.cat(feature_blocks, dim=1)
    features = (features - features.mean(dim=0, keepdim=True)) / (features.std(dim=0, keepdim=True) + 1e-6)
    return features


def load_graph_coloring_data_from_file(file_path: str, feature_dim: int) -> Data:
    """Load one graph coloring instance from disk and convert to PyG Data."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Graph file not found: {file_path}")

    graph = _read_graph_from_txt(file_path)
    pyg_data = from_networkx(graph)
    pyg_data.edge_index = pyg_data.edge_index.long()
    pyg_data.x = _create_graph_coloring_features(graph, feature_dim)
    pyg_data.num_nodes = graph.number_of_nodes()
    pyg_data.graph_name = os.path.splitext(os.path.basename(file_path))[0]
    pyg_data.file_path = file_path
    return pyg_data


def load_graph_coloring_graphs_from_directory(directory: str,
                                              prefixes: Optional[List[str]],
                                              feature_dim: int):
    """Load all matching graph coloring graphs from a directory."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Graph directory does not exist: {directory}")

    if prefixes:
        files = calc_txt_files_with_prefixes(directory, prefixes)
    else:
        files = [
            os.path.join(directory, f)
            for f in sorted(os.listdir(directory))
            if f.endswith(".txt")
        ]

    if not files:
        print(f"[PIGNN] No graph files found under {directory} with prefixes={prefixes}.")
        return []

    graphs = []
    failed_files: List[tuple[str, str]] = []
    for path in files:
        try:
            graphs.append(load_graph_coloring_data_from_file(path, feature_dim))
        except Exception as exc:  # pylint: disable=broad-except
            failed_files.append((path, str(exc)))

    if failed_files:
        preview = ", ".join(os.path.basename(p) for p, _ in failed_files[:3])
        print(f"[PIGNN] Skipped {len(failed_files)} files due to errors. Examples: {preview}")
    return graphs
