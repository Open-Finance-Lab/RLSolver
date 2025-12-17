import os
import numpy as np
from config import transfer_nxgraph_to_local_dict
from rlsolver.methods.util_read_data import read_nxgraph
from rlsolver.methods.util import calc_txt_files_with_prefixes

def welsh_powell(graph):
    graph = transfer_nxgraph_to_local_dict(graph)
    sorted_vertices = sorted(graph, key=lambda v: len(graph[v]), reverse=True)
    color_graph = {}
    color = 1
    for vertex in sorted_vertices:
        if vertex not in color_graph:
            color_graph[vertex] = (color)
            neighbors=set(graph[vertex])
            for other_vertex in sorted_vertices:
                if (
                    other_vertex not in color_graph
                    and other_vertex not in neighbors
                ):
                    color_graph[other_vertex] = color
                    neighbors.update(graph[other_vertex])
            color += 1
    colors = []
    keys = sorted(color_graph.keys())
    for key in keys:
        colors.append(color_graph[key])
    return colors

if __name__ == '__main__':
    dir = '../../data/syn_BA'
    prefixes = ['BA_5_']
    files = calc_txt_files_with_prefixes(dir, prefixes)
    for i in range(len(files)):
        filename = files[i]
        graph = read_nxgraph(filename)
        result = welsh_powell(graph)
        print(result)
    pass