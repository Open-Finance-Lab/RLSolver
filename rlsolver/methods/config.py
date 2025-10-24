import torch as th
from typing import List
from enum import Enum, unique
import os
from rlsolver.methods.L2A.graph_utils import GraphList, obtain_num_nodes

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

# DATA_FILENAME = "../data/gset/BA_100_ID0.txt"  # one instance
DIRECTORY_DATA = "../data/syn_BA"  # used in multi instances
PREFIXES = ["BA_100_ID0"]  # used in multi instances - 匹配所有BA图(100到1000节点)

DEVICE: th.device = calc_device(GPU_ID)

GRAPH_TYPE = GraphType.PL
GRAPH_TYPES: List[GraphType] = [GraphType.ER, GraphType.PL, GraphType.BA]
    # graph_types = ["erdos_renyi", "powerlaw", "barabasi_albert"]
NUM_IDS = 30  # ID0, ..., ID29


INF = 1e6

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


ModelDir = './model'  # FIXME plan to cancel
class ConfigPolicy:
    def __init__(self, graph_list: GraphList = None, graph_type: str = 'max_cut', num_nodes: int = 0):
        num_nodes = num_nodes if num_nodes > 0 else obtain_num_nodes(graph_list=graph_list)

        self.graph_type = graph_type
        self.num_nodes = num_nodes

        '''train'''
        self.num_sims = 2 ** 3  # LocalSearch 的初始解数量
        self.num_repeats = 2 ** 4  # LocalSearch 对于每以个初始解进行复制的数量
        self.num_searches = 2 ** 2  # LocalSearch 添加噪声的次数
        self.reset_gap = 2 ** 5  # 重置并开始新的搜索需要的迭代步数
        self.num_iters = 2 ** 7  # 进行迭代搜索的总步数
        self.num_sgd_steps = 2 ** 2  # 每一次根据监督信号进行梯度下降的次数
        self.entropy_weight = 4  # 策略熵的权重（退火方案会控制策略熵，在一个周期内由entropy_weight*1变化到0）

        self.learning_rate = 2 ** -10  # 优化器的学习率
        self.weight_decay = 0  # 2 ** -16 # 优化器的权重衰减
        self.net_path = f"{ModelDir}/policy_net_{graph_type}_Node{num_nodes}.pth"  # policy_net的保存路径

        self.show_gap = 2 ** 2  # 训练时，打印训练进度的间隔步数

        '''model'''
        self.num_heads = 8
        self.num_layers = 4
        self.mid_dim = 256
        self.inp_dim = num_nodes  # 输入是邻接矩阵
        self.out_dim = 1  # 输出是节点对应的概率
        sqrt_num_nodes = int(num_nodes ** 0.5)
        self.embed_dim = max(sqrt_num_nodes - sqrt_num_nodes % self.num_heads, 32)  # 编码后的节点嵌入向量的长度

    def load_net(self, net_path: str = '', device=None, if_valid: bool = False):
        import torch as th
        net_path = net_path if net_path else self.net_path
        device = device if device else th.device('cpu')

        # from network import PolicyRNN
        # net = PolicyRNN(inp_dim=self.inp_dim, mid_dim=self.mid_dim, out_dim=self.out_dim,
        #                 embed_dim=self.embed_dim, num_heads=self.num_heads, num_layers=self.num_layers).to(device)
        from rlsolver.methods.L2A.network import PolicyTRS
        net = PolicyTRS(inp_dim=self.inp_dim, mid_dim=self.mid_dim, out_dim=self.out_dim,
                        embed_dim=self.embed_dim, num_heads=self.num_heads, num_layers=self.num_layers).to(device)
        if if_valid:
            if not not os.path.isfile(net_path):
                raise FileNotFoundError(f"| ConfigPolicy.load_net()  net_path {net_path}")
            net.load_state_dict(th.load(net_path, map_location=device))
        else:  # if_train
            pass
        net = th.compile(net) if th.__version__ < '2.0' else net
        return net