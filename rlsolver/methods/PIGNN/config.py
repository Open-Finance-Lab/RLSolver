import torch as th
from typing import List
from enum import Enum, unique
import os


@unique
class MODEL_KEY_DICT(Enum):
    GCN: str = "GCN" # 0
    GAT: str = "GAT" # 1
    GATv2: str = "GATv2" # 2
    GraphConv: str = "GraphConv" # 3

from rlsolver.methods.config import PROBLEM, Problem
PROBLEM = Problem.maxcut
assert PROBLEM in [Problem.maxcut, Problem.MIS, Problem.graph_coloring]

SEED: int = 42
NUM_GRAPHS: int = 100
NUM_NODES: int = 100
NODE_DEGREE: int = 3
BATCH_SIZE: int = 1
LEARNING_RATE: float = 1e-4
EPOCHS: int = int(1e5)
NUM_WORKERS: int = 6
GPU_NUM: int = 0
GNN_MODEL = MODEL_KEY_DICT.GCN
NUM_HEADS: int = 4 # Num of heads if you wish to use GAT Ansatz

# Graph Coloring specific parameters
NUM_COLORS: int = 6  # Number of colors for graph coloring problem
LAMBDA_ENTROPY: float = 0.1  # Entropy regularization coefficient
LAMBDA_BALANCE: float = 0.05  # Color balance regularization coefficient

# Additional Graph Coloring parameters
IN_DIM: int = 16  # Feature dimension for graph coloring
TEMPERATURE: float = 1.2  # Temperature sampling parameter for decoding
TRIALS: int = 10  # Number of sampling trials for temperature sampling

