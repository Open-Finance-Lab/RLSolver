from enum import Enum
import networkx as nx
import time
import os
import sys
cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))
from rlsolver.methods.util_read_data import read_nxgraph
from rlsolver.methods.util import calc_txt_files_with_prefixes

class Alg(Enum):
    degree_of_saturation = 'degree_of_saturation' # DSATUR
    greedy = 'greedy'
    recursive = 'recursive'
    welsh_powell = 'welsh_powell'

ALG = Alg.degree_of_saturation

def transfer_nxgraph_to_local_dict(graph: nx.Graph):
    res = {}
    for node, edges in graph.adjacency():
        res[node] = list(edges.keys())
    return res

def transfer_nxgraph_to_local_list(graph: nx.Graph):
    res = [0] * graph.number_of_nodes()
    for node, edges in graph.adjacency():
        res[node] = list(edges.keys())
    return res

if __name__ == '__main__':
    dir = '../../data/syn_BA'
    prefixes = ['BA_5_']
    files = calc_txt_files_with_prefixes(dir, prefixes)
    for i in range(len(files)):
        start_time = time.time()
        filename = files[i]
        print(f'Start the {i}-th file: {filename}')
        graph = read_nxgraph(filename)
        loal = transfer_nxgraph_to_local_dict(graph)
        print('loal graph: ', loal)