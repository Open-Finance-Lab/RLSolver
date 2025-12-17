import os
import numpy as np
from config import transfer_nxgraph_to_local_dict
from rlsolver.methods.util_read_data import read_nxgraph
from rlsolver.methods.util import calc_txt_files_with_prefixes
def greedy(graph, order=None):
    graph = transfer_nxgraph_to_local_dict(graph)
    n = len(graph)
    color = [-1] * n
    if order is None:
        order = list(range(n))  # 默认顺序
    for u in order:
        used_colors = set()
        for v in graph[u]:
            if color[v] != -1:
                used_colors.add(color[v])
        for c in range(1, n + 1):
            if c not in used_colors:
                color[u] = c
                break
    return color



if __name__ == '__main__':
    dir = '../../data/syn_BA'
    prefixes = ['BA_5_']
    files = calc_txt_files_with_prefixes(dir, prefixes)
    for i in range(len(files)):
        filename = files[i]
        graph = read_nxgraph(filename)
        result = greedy(graph)
        print(result)
    pass
