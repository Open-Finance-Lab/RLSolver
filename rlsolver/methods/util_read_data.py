import sys
import os
import plotly.io as pio
import plotly.graph_objects as go
import numpy  as np
import random
import networkx as nx
import random

from numpy.core.defchararray import isnumeric
cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

from typing import List, Tuple, Union
from rlsolver.methods.config import GRAPH_TYPE
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import networkx as nx
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import torch as th
import random
from rlsolver.methods.util import calc_num_nodes_in_mygraph
from rlsolver.methods.util import build_adjacency_bool
from rlsolver.methods.util import get_hot_image_of_graph
from rlsolver.methods.util import show_array2d
from rlsolver.methods.config import MyGraph
from rlsolver.methods.config import MyNeighbor
from rlsolver.methods.config import DIRECTORY_DATA
from rlsolver.methods.config import GRAPH_TYPES
from rlsolver.methods.util_generate import generate_mygraph

from rlsolver.methods.config import MyGraph
from rlsolver.methods.config import MyNeighbor
GraphTypes = ['BA', 'ER', 'PL']
TEN = th.Tensor

from rlsolver.methods.util import calc_txt_files_with_prefixes
from rlsolver.methods.util_generate import generate_mygraph

# read graph file, e.g., gset_14.txt, as networkx.Graph
# The nodes in file start from 1, but the nodes start from 0 in our codes.
def read_nxgraph(filename: str) -> nx.Graph():
    graph = nx.Graph()
    with open(filename, 'r') as file:
        # lines = []
        line = file.readline()
        is_first_line = True
        while line is not None and line != '':
            if '//' not in line:
                if is_first_line:
                    strings = line.split(" ")
                    num_nodes = int(strings[0])
                    num_edges = int(strings[1])
                    nodes = list(range(num_nodes))
                    graph.add_nodes_from(nodes)
                    is_first_line = False
                else:
                    node1, node2, weight = line.split(" ")
                    graph.add_edge(int(node1) - 1, int(node2) - 1, weight=weight)
            line = file.readline()
    return graph

def read_nxgraphs(directory: str, prefixes: List[str]) -> List[nx.Graph]:
    graphs = []
    files = calc_txt_files_with_prefixes(directory, prefixes)
    for i in range(len(files)):
        filename = files[i]
        graph = read_nxgraph(filename)
        graphs.append(graph)
    return graphs

def read_mygraph(filename: str) -> MyGraph:
    with open(filename, 'r') as file:
        lines = file.readlines()
        lines = [[int(i1) for i1 in i0.split()] for i0 in lines]
    num_nodes, num_edges = lines[0]
    mygraph = [(n0 - 1, n1 - 1, dt) for n0, n1, dt in lines[1:]]  # 将node_id 由“从1开始”改为“从0开始”
    return mygraph

def load_mygraph(DataDir, graph_name: str):
    graph_types = GRAPH_TYPES
    if os.path.exists(f"{DataDir}/{graph_name}.txt"):
        txt_path = f"{DataDir}/{graph_name}.txt"
        graph = read_mygraph(txt_path)
    elif graph_name.split('_')[0] in graph_types and len(graph_name.split('_')) == 3:
        graph_type, num_nodes, valid_i = graph_name.split('_')
        num_nodes = int(num_nodes)
        valid_i = int(valid_i[len('ID'):])
        random.seed(valid_i)
        graph, _, _ = generate_mygraph(graph_type, num_nodes)
        random.seed()
    elif graph_name.split('_')[0] in graph_types and len(graph_name.split('_')) == 2:
        graph_type, num_nodes = graph_name.split('_')
        num_nodes = int(num_nodes)
        graph, _, _= generate_mygraph(graph_type, num_nodes)
    elif os.path.isfile(graph_name):
        txt_path = graph_name
        graph = read_mygraph(txt_path)
    else:
        raise ValueError(f"DataDir {DataDir} | graph_name {graph_name}")
    return graph

def load_mygraph2(dataDir='./data/syn_'+GRAPH_TYPE.value, graph_name: str= ""):
    if os.path.exists(f"{dataDir}/{graph_name}.txt"):
        txt_path = f"{dataDir}/{graph_name}.txt"
        mygraph = read_mygraph(filename=txt_path)
    elif os.path.isfile(graph_name) and os.path.splitext(graph_name)[-1] == '.txt':
        txt_path = graph_name
        mygraph = read_mygraph(filename=txt_path)
    elif GRAPH_TYPE and graph_name.find('ID') == -1:
        num_nodes = int(graph_name.split('_')[-1])
        mygraph, _, _ = generate_mygraph(num_nodes=num_nodes, graph_type=GRAPH_TYPE)
    elif GRAPH_TYPE and graph_name.find('ID') >= 0:
        num_nodes, valid_i = graph_name.split('_')[-2:]
        num_nodes = int(num_nodes)
        valid_i = int(valid_i[len('ID'):])
        random.seed(valid_i)
        mygraph, _, _ = generate_mygraph(num_nodes=num_nodes, graph_type=GRAPH_TYPE)
        random.seed()
    else:
        raise ValueError(f"DataDir {dataDir} | graph_name {graph_name} txt_path {dataDir}/{graph_name}.txt")
    return mygraph



def build_adjacency_indies(mygraph: MyGraph, if_bidirectional: bool = False) -> (MyNeighbor, MyNeighbor):
    """
    用二维列表list2d表示这个图：
    [
        [1, 2],
        [],
        [3],
        [],
    ]
    其中：
    - list2d[0] = [1, 2]
    - list2d[2] = [3]

    对于稀疏的矩阵，可以直接记录每条边两端节点的序号，用shape=(2,N)的二维列表 表示这个图：
    0, 1
    0, 2
    2, 3
    如果条边的长度为1，那么表示为shape=(2,N)的二维列表，并在第一行，写上 4个节点，3条边的信息，帮助重建这个图，然后保存在txt里：
    4, 3
    0, 1, 1
    0, 2, 1
    2, 3, 1
    """
    num_nodes = calc_num_nodes_in_mygraph(mygraph=mygraph)

    n0_to_n1s = [[] for _ in range(num_nodes)]  # 将 node0_id 映射到 node1_id
    n0_to_dts = [[] for _ in range(num_nodes)]  # 将 mode0_id 映射到 node1_id 与 node0_id 的距离
    for n0, n1, distance in mygraph:
        n0_to_n1s[n0].append(n1)
        n0_to_dts[n0].append(distance)
        if if_bidirectional:
            n0_to_n1s[n1].append(n0)
            n0_to_dts[n1].append(distance)
    n0_to_n1s = [th.tensor(node1s) for node1s in n0_to_n1s]
    n0_to_dts = [th.tensor(node1s) for node1s in n0_to_dts]
    assert num_nodes == len(n0_to_n1s)
    assert num_nodes == len(n0_to_dts)

    '''sort'''
    for i, node1s in enumerate(n0_to_n1s):
        sort_ids = th.argsort(node1s)
        n0_to_n1s[i] = n0_to_n1s[i][sort_ids]
        n0_to_dts[i] = n0_to_dts[i][sort_ids]
    return n0_to_n1s, n0_to_dts


def update_xs_by_vs(xs0: TEN, vs0: TEN, xs1: TEN, vs1: TEN, if_maximize: bool) -> int:
    """
    并行的子模拟器数量为 num_sims, 解x 的节点数量为 num_nodes
    xs: 并行数量个解x,xs.shape == (num_sims, num_nodes)
    vs: 并行数量个解x对应的 objective value. vs.shape == (num_sims, )

    更新后，将xs1，vs1 中 objective value数值更高的解x 替换到xs0，vs0中
    如果被更新的解的数量大于0，将返回True
    """
    good_is = vs1.ge(vs0) if if_maximize else vs1.le(vs0)
    xs0[good_is] = xs1[good_is]
    vs0[good_is] = vs1[good_is]
    return good_is.shape[0]

def pick_xs_by_vs(xs: TEN, vs: TEN, num_repeats: int, if_maximize: bool) -> (TEN, TEN):
    # update good_xs: use .view() instead of .reshape() for saving GPU memory
    num_nodes = xs.shape[1]
    num_sims = xs.shape[0] // num_repeats

    xs_view = xs.view(num_repeats, num_sims, num_nodes)
    vs_view = vs.view(num_repeats, num_sims)
    ids = vs_view.argmax(dim=0) if if_maximize else vs_view.argmin(dim=0)

    sim_ids = th.arange(num_sims, device=xs.device)
    good_xs = xs_view[ids, sim_ids]
    good_vs = vs_view[ids, sim_ids]
    return good_xs, good_vs

# def read_set_cover(filename: str):
#     with open(filename, 'r') as file:
#         # lines = []
#         line = file.readline()
#         item_matrix = []
#         while line is not None and line != '':
#             if 'p set' in line:
#                 strings = line.split(" ")
#                 num_items = int(strings[-2])
#                 num_sets = int(strings[-1])
#             elif 's' in line:
#                 strings = line.split(" ")
#                 items = [int(s) for s in strings[1:]]
#                 item_matrix.append(items)
#             else:
#                 raise ValueError("error in read_set_cover")
#             line = file.readline()
#     return num_items, num_sets, item_matrix

def read_knapsack_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        N, W = map(int, lines[0].split())
        items = []
        for line in lines[1:]:
            weight, value = map(int, line.split())
            items.append((weight, value))
    return N, W, items


def read_set_cover_data(filename):
    with open(filename, 'r') as file:
        first_line = file.readline()
        total_elements, total_subsets = map(int, first_line.split())
        subsets = []
        for line in file:
            subset = list(map(int, line.strip().split()))
            subsets.append(subset)

    return total_elements, total_subsets, subsets


def read_tsp_file(filename: str):
    coordinates = np.array([])
    prev_index = None
    with open(filename, 'r') as file:
        count = 0
        while True:
            line = file.readline()
            count += 1
            if 'EOF' in line:
                break
            parts = line.split(' ')
            new_parts = [i for i in parts if len(i) > 0 and i != '\n']
            if len(new_parts) == 3 and isnumeric(new_parts[0]):
                index_str, x_str, y_str = new_parts
                index = int(index_str)
                if (prev_index is None and index == 1) or (prev_index is not None and index == prev_index + 1):
                    if (prev_index is None and index == 1) and len(coordinates) > 0:
                        coordinates = np.array([])
                        continue
                    x = float(x_str)
                    y = float(y_str)
                    if len(coordinates) == 0:
                        coordinates = np.array([(x, y)])
                    else:
                        coordinates = np.append(coordinates, [(x, y)], axis=0)
                    prev_index = index
    assert index == len(coordinates)
    num_nodes = len(coordinates)
    nodes = list(range(num_nodes))
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            xi, yi = coordinates[i]
            xj, yj = coordinates[j]
            dist = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            graph.add_edge(i, j, weight=dist)
    return graph, coordinates

def check_get_hot_tenor_of_graph():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    if_save = True

    # graph_names = []
    # for graph_type in GraphTypes:
    #     for num_nodes in (100, 101):
    #         for seed_id in range(1):
    #             graph_names.append(f'{graph_type}_{num_nodes}_ID{seed_id}')
    DataDir = "../../data/syn_BA"
    graph_names = ["BA_100_ID0"]
    for graph_name in graph_names:
        mygraph: MyGraph = load_mygraph2(dataDir=DataDir, graph_name=graph_name)

        graph = nx.Graph()
        for n0, n1, weight in mygraph:
            graph.add_edge(n0, n1, weight=weight)

        for hot_type in ('avg', 'sum'):
            adj_bool = build_adjacency_bool(mygraph=mygraph, if_bidirectional=True).to(device)
            hot_array = get_hot_image_of_graph(adj_bool=adj_bool, hot_type=hot_type).cpu().data.numpy()
            title = f"{hot_type}_{graph_name}_N{graph.number_of_nodes()}_E{graph.number_of_edges()}"
            show_array2d(ary=hot_array, title=title, if_save=if_save)
            print(f"title {title}")

    print()
if __name__ == '__main__':

    read_txt = True
    if read_txt:
        graph1 = read_nxgraph('../data/gset/gset_14.txt')
        graph2 = read_nxgraph('../data/syn_BA/BA_100_ID0.txt')

    check_get_hot_tenor_of_graph()