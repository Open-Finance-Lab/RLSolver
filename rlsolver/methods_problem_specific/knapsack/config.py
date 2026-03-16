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
    branch_bound = 'branch_bound' # DSATUR
    exhaustive_search = 'exhaustive_search'
    dynamic_programming = 'dynamic_programming'
    fptas = 'fptas'
    greedy = 'greedy'
    simulated_annealing = 'simulated_annealing'

ALG = Alg.greedy


if __name__ == '__main__':
    dir = '../../data/syn_BA'
    prefixes = ['BA_5_']
    files = calc_txt_files_with_prefixes(dir, prefixes)
    for i in range(len(files)):
        start_time = time.time()
        filename = files[i]
        print(f'Start the {i}-th file: {filename}')
        graph = read_nxgraph(filename)
        
