import torch as th
from typing import Iterable, Optional, List, Union
from enum import Enum, unique
import os


@unique
class MODEL_KEY_DICT(Enum):
    GCN: str = "GCN"  # 0
    GAT: str = "GAT"  # 1
    GATv2: str = "GATv2"  # 2
    GraphConv: str = "GraphConv"  # 3

from rlsolver.methods.config import Problem

# Mode control (referencing ECO_S2V pattern)
TRAIN_INFERENCE: int = 0  # 0: train mode, 1: inference mode

# Problem type selection
PROBLEM: Problem = Problem.maxcut
assert PROBLEM in [Problem.maxcut, Problem.MIS, Problem.graph_coloring]

# ================== Mode-specific Configurations ==================

# Training specific configurations
TRAIN_SEED: int = 42
TRAIN_NUM_GRAPHS: int = 100
TRAIN_NUM_NODES: int = 100
TRAIN_BATCH_SIZE: int = 1
TRAIN_EPOCHS: int = int(1e5)
TRAIN_NUM_WORKERS: int = 6
TRAIN_GPU_NUM: int = 0
TRAIN_LEARNING_RATE: float = 1e-4

# Inference specific configurations
INFERENCE_SEED: int = 42
INFERENCE_BATCH_SIZE: int = 32
INFERENCE_GPU_NUM: int = 0
INFERENCE_NUM_WORKERS: int = 4
INFERENCE_NUM_NODES: int = 100  # or [50, 100, 200] for multiple sizes

# Model checkpoint and paths
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CONFIG_DIR, "../../.."))
MODEL_CHECKPOINT_DIR: str = os.path.join(_PROJECT_ROOT, "rlsolver", "trained_agent")
MODEL_FILENAME_TEMPLATE: str = "PIGNN_{problem}_{graph}_{nodes}_{timestamp}.pth"
RESULTS_DIR: str = os.path.join(_PROJECT_ROOT, "rlsolver", "result")
SAVE_MODEL: bool = True

SEED: int = TRAIN_SEED
NUM_GRAPHS: int = TRAIN_NUM_GRAPHS
NUM_NODES: int = TRAIN_NUM_NODES
NODE_DEGREE: int = 3
BATCH_SIZE: int = TRAIN_BATCH_SIZE
LEARNING_RATE: float = TRAIN_LEARNING_RATE
EPOCHS: int = TRAIN_EPOCHS
NUM_WORKERS: int = TRAIN_NUM_WORKERS
GPU_NUM: int = TRAIN_GPU_NUM  # Mirrors TRAIN_GPU_NUM (-1 for CPU, >=0 for GPU ID)
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

# Dataset-driven inference parameters for Graph Coloring
GRAPH_COLORING_INFERENCE_USE_DATASET: bool = False
GRAPH_COLORING_INFERENCE_DATA_DIR: str = os.path.join(_PROJECT_ROOT, "rlsolver", "data")
GRAPH_COLORING_INFERENCE_PREFIXES: List[str] = []
GRAPH_COLORING_INFERENCE_WRITE_RESULTS: bool = True
GRAPH_COLORING_INFERENCE_ALG_NAME: str = "PIGNN_graph_coloring"

# ================== Model Naming and Management ==================


def generate_model_filename(problem: str,
                            num_nodes: int,
                            graph_type: str = "BA",
                            timestamp: Optional[str] = None) -> str:
    """
    Generate standardized model filename following project conventions.

    Args:
        problem: Problem type (maxcut, MIS, graph_coloring)
        num_nodes: Number of nodes in the training graphs
        graph_type: Type of graphs used (BA, ER, PL, etc.)
        timestamp: Optional timestamp for version control

    Returns:
        Standardized model filename
    """
    from datetime import datetime

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d")

    return MODEL_FILENAME_TEMPLATE.format(
        problem=problem,
        graph=graph_type,
        nodes=num_nodes,
        timestamp=timestamp,
    )


def get_model_save_path(problem: Optional[str] = None,
                        num_nodes: Optional[int] = None,
                        graph_type: str = "BA",
                        timestamp: Optional[str] = None) -> str:
    """
    Get the full path for saving a model file.

    Args:
        problem: Problem type (maxcut, MIS, graph_coloring)
        num_nodes: Number of nodes in the training graphs
        graph_type: Type of graphs used (BA, ER, PL, etc.)
        timestamp: Optional timestamp for version control

    Returns:
        Full path for model file
    """
    if problem is None:
        problem = PROBLEM.name.lower() if PROBLEM else "unknown"
    if num_nodes is None:
        num_nodes = TRAIN_NUM_NODES if TRAIN_INFERENCE == 0 else INFERENCE_NUM_NODES

    filename = generate_model_filename(problem, num_nodes, graph_type, timestamp)
    return os.path.join(MODEL_CHECKPOINT_DIR, filename)


def _iter_num_nodes(num_nodes: Optional[Union[Iterable[int], int]]):
    if num_nodes is None:
        yield TRAIN_NUM_NODES if TRAIN_INFERENCE == 0 else INFERENCE_NUM_NODES
        return
    if isinstance(num_nodes, Iterable) and not isinstance(num_nodes, (str, bytes)):
        for value in num_nodes:
            yield value
    else:
        yield int(num_nodes)


def get_model_load_path(problem: str,
                        num_nodes: Optional[Union[Iterable[int], int]] = None,
                        graph_type: str = "BA",
                        timestamp: Optional[str] = None) -> str:
    """
    Get the full path for loading a model file.

    Args:
        problem: Problem type (maxcut, MIS, graph_coloring)
        num_nodes: Number of nodes in the training graphs
        graph_type: Type of graphs used (BA, ER, PL, etc.)
        timestamp: Optional timestamp for specific version

    Returns:
        Full path for model file
    """
    if problem is None:
        problem = PROBLEM.name.lower() if PROBLEM else "unknown"

    candidate_paths = []
    if timestamp is None:
        import glob

    for node_count in _iter_num_nodes(num_nodes):
        if timestamp is None:
            pattern = os.path.join(
                MODEL_CHECKPOINT_DIR,
                f"PIGNN_{problem}_{graph_type}_{node_count}_*.pth",
            )
            matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
            candidate_paths.extend(matches)
        else:
            filename = generate_model_filename(problem, node_count, graph_type, timestamp)
            candidate_paths.append(os.path.join(MODEL_CHECKPOINT_DIR, filename))

    for path in candidate_paths:
        if os.path.exists(path):
            return path

    legacy_path = os.path.join(MODEL_CHECKPOINT_DIR, f"{problem.lower()}_model.pth")
    return legacy_path
