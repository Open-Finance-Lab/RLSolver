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
