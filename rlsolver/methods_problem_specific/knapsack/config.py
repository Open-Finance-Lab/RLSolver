from enum import Enum
import networkx as nx
import time
from rlsolver.methods.util_read_data import read_nxgraph
from rlsolver.methods.util import calc_txt_files_with_prefixes

class Alg(Enum):
    branch_bound = 'branch_bound' # DSATUR
    brute_force = 'brute_force'
    dynamic_programming = 'dynamic_programming'
    fptas = 'fptas'
    greedy = 'greedy'
    simulated_annealing = 'simulated_annealing'

ALG = Alg.brute_force


if __name__ == '__main__':
    dir = '../../data/syn_BA'
    prefixes = ['BA_5_']
    files = calc_txt_files_with_prefixes(dir, prefixes)
    for i in range(len(files)):
        start_time = time.time()
        filename = files[i]
        print(f'Start the {i}-th file: {filename}')
        graph = read_nxgraph(filename)
        
