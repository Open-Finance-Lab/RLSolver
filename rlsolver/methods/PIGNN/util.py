import math
import torch

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

def eval_graph_coloring(edge_index, pred, d, n, num_colors):
    """
    Evaluate graph coloring solution.

    Args:
        edge_index: Tensor of shape [2, num_edges] representing graph connectivity
        pred: Tensor of shape [num_nodes, num_colors] with softmax probabilities
        d: Average node degree (not used in current implementation)
        n: Number of nodes
        num_colors: Number of available colors

    Returns:
        coloring_energy: Negative of conflict-free score (higher is better)
        chromatic_ratio: Ratio of used colors to available colors
    """
    # Get color assignments by taking argmax
    color_assignments = torch.argmax(pred, dim=1)

    # Count violations (adjacent nodes with same color)
    i, j = edge_index
    violations = torch.sum(color_assignments[i] == color_assignments[j]).float()

    # Count unique colors used
    used_colors = color_assignments.unique().numel()

    # Energy is negative violations (we want to minimize violations)
    coloring_energy = -violations

    # Chromatic ratio: lower is better (closer to 1 means using all available colors efficiently)
    chromatic_ratio = used_colors / num_colors

    return coloring_energy, torch.tensor(chromatic_ratio)
