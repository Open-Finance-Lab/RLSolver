import networkx as nx
import heapq
from config import transfer_nxgraph_to_local_dict
from rlsolver.methods.util_read_data import read_nxgraph
from rlsolver.methods.util import calc_txt_files_with_prefixes

def recursive(graph, max_colors=None):
    max_colors = graph.number_of_nodes() if max_colors is None else max_colors
    graph = transfer_nxgraph_to_local_dict(graph)
    n = len(graph)
    color_assignment = [-1] * n
    adjacency_list = graph

    def is_safe(v, c):
        return all(color_assignment[u] != c for u in adjacency_list[v])

    def backtrack(v):
        if v == n:
            return True
        for c in range(1, max_colors + 1):
            if is_safe(v, c):
                color_assignment[v] = c
                if backtrack(v + 1):
                    return True
                color_assignment[v] = -1
        return False

    if backtrack(0):
        return color_assignment
    else:
        return None


if __name__ == '__main__':
    dir = '../../data/syn_BA'
    prefixes = ['BA_5_']
    files = calc_txt_files_with_prefixes(dir, prefixes)
    for i in range(len(files)):
        filename = files[i]
        graph = read_nxgraph(filename)
        result = recursive(graph)
        print(result)
    pass