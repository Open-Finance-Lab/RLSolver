import torch as th
from typing import List, Tuple
from enum import Enum, unique

GraphList = List[Tuple[int, int, int]]  # 每条边两端点的索引以及边的权重 List[Tuple[Node0ID, Node1ID, WeightEdge]]
IndexList = List[List[int]]  # 按索引顺序记录每个点的所有邻居节点 IndexList[Node0ID] = [Node1ID, ...]

@unique
class GraphType(Enum):
    BA: str = "BA"  # "barabasi_albert"
    ER: str = "ER"  # "erdos_renyi"
    PL: str = "PL"  # "powerlaw"

def calc_device(gpu_id: int):
    return th.device(f"cuda:{gpu_id}" if th.cuda.is_available() and gpu_id >= 0 else "cpu")

@unique
class Problem(Enum):
    maxcut = "maxcut"
    graph_partitioning = "graph_partitioning"
    number_partitioning = "number_partitioning"
    MVC = "MVC" # minimum_vertex_cover
    BILP = "BILP"
    MIS = "MIS" # maximum_independent_set
    knapsack = "knapsack"
    set_cover = "set_cover"
    graph_coloring = "graph_coloring"
    portfolio_allocation = "portfolio_allocation"
    TNCO = "TNCO"
    VRP = "VRP"
    TSP = "TSP"
PROBLEM = Problem.MIS

GPU_ID: int = 0  # -1: cpu, >=0: gpu

DATA_FILENAME = "../data/gset/BA_100_ID0.txt"  # one instance
DIRECTORY_DATA = "../data/syn_BA"  # used in multi instances
PREFIXES = ["BA_100_ID0"]  # used in multi instances - 匹配所有BA图(100到1000节点)

DEVICE: th.device = calc_device(GPU_ID)

GRAPH_TYPE = GraphType.PL
GRAPH_TYPES: List[GraphType] = [GraphType.ER, GraphType.PL, GraphType.BA]
    # graph_types = ["erdos_renyi", "powerlaw", "barabasi_albert"]
NUM_IDS = 30  # ID0, ..., ID29


INF = 1e6

# train or inference
TRAIN_INFERENCE = 1  # 0: train, 1: inference
assert TRAIN_INFERENCE in [0, 1]


# training
TRAIN_GPU_ID = 0
TRAIN_DEVICE = calc_device(TRAIN_GPU_ID)
NUM_TRAIN_NODES = 20
NUM_TRAIN_ENVS = 2 ** 8


# inference
INFERENCE_GPU_ID = 0
INFERENCE_DEVICE = calc_device(INFERENCE_GPU_ID)
NUM_TRAINED_NODES_IN_INFERENCE = 20  # also used in select_best_neural_network
NUM_INFERENCE_NODES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 2000, 3000, 4000, 5000, 10000]
INFERENCE_PREFIXES = [GRAPH_TYPE.value + "_" + str(i) + "_" for i in NUM_INFERENCE_NODES]
# PREFIXES = ["BA_100_", "BA_200_", "BA_300_", "BA_400_", "BA_500_""]  # Replace with your desired prefixes
NUM_INFERENCE_ENVS = 50




# RUNNING_DURATIONS = [600, 1200, 1800, 2400, 3000, 3600]  # store results
RUNNING_DURATIONS = [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600]  # store results

# None: write results when finished.
# others: write results in mycallback. seconds, the interval of writing results to txt files
GUROBI_INTERVAL = None  # None: not using interval, i.e., not using mycallback function, write results when finished. If not None such as 100, write result every 100s
GUROBI_TIME_LIMITS = [1 * 3600]  # seconds
# GUROBI_TIME_LIMITS = [600, 1200, 1800, 2400, 3000, 3600]  # seconds
# GUROBI_TIME_LIMITS2 = list(range(10 * 60, 1 * 3600 + 1, 10 * 60))  # seconds
GUROBI_VAR_CONTINUOUS = False  # True: relax it to LP, and return x values. False: sovle the primal MILP problem
GUROBI_MILP_QUBO = 1  # 0: MILP, 1: QUBO. default: QUBO, since using QUBO is generally better than MILP.
assert GUROBI_MILP_QUBO in [0, 1]

