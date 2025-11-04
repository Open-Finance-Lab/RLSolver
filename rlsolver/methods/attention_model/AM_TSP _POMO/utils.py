# utils.py

import numpy as np
import torch

try:
    import elkai
    ELKAI_AVAILABLE = True
except ImportError:
    ELKAI_AVAILABLE = False


def get_heuristic_solution(pointset, scale=100000.0):
    """Get heuristic solution using elkai (LKH algorithm).
    
    Args:
        pointset: Tensor or numpy array of shape [num_nodes, 2]
        scale: Scaling factor for coordinates
        
    Returns:
        tour_length: Length of the heuristic tour
    """
    if not ELKAI_AVAILABLE:
        return None
    if isinstance(pointset, (torch.Tensor, torch.cuda.FloatTensor)):
        pointset = pointset.detach().cpu().numpy()
    num_points = len(pointset)
    dist_matrix = np.zeros((num_points, num_points), dtype=np.int32)
    for i in range(num_points):
        for j in range(i + 1, num_points):
            dist = int(np.linalg.norm(pointset[i] - pointset[j]) * scale)
            dist_matrix[i, j] = dist_matrix[j, i] = dist
    tour = elkai.solve_int_matrix(dist_matrix)
    tour_length = 0.0
    for i in range(num_points):
        tour_length += dist_matrix[tour[i], tour[(i + 1) % num_points]]
    return tour_length / scale


def clip_grad_norm(parameters, max_norm):
    """Clip gradients by norm."""
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm)


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
