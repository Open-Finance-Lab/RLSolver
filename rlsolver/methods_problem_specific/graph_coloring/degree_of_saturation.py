import networkx as nx
import heapq
from config import transfer_nxgraph_to_local_dict
from rlsolver.methods.util_read_data import read_nxgraph
from rlsolver.methods.util import calc_txt_files_with_prefixes

def degree_of_saturation(graph):
    graph = transfer_nxgraph_to_local_dict(graph)
    # graph = {
    #     0: [1, 2],
    #     1: [0, 2, 3],
    #     2: [0, 1],
    #     3: [1]
    # }
    n = len(graph)
    color = [-1] * n
    saturation = [0] * n  # 饱和度：已用颜色种类数
    degree = [len(graph[i]) for i in range(n)]
    uncolored = set(range(n))

    # 使用最大堆管理饱和度
    heap = [(-saturation[i], -degree[i], i) for i in range(n)]
    heapq.heapify(heap)

    while uncolored:
        _, _, u = heapq.heappop(heap)
        if color[u] != -1:
            continue
        # 获取邻居颜色集合
        used_colors = {color[v] for v in graph[u] if color[v] != -1}
        for c in range(1, n + 1):
            if c not in used_colors:
                color[u] = c
                break
        uncolored.remove(u)
        # 更新邻居饱和度
        for v in graph[u]:
            if v in uncolored:
                new_sat = len({color[w] for w in graph[v] if color[w] != -1})
                saturation[v] = new_sat
                heapq.heappush(heap, (-new_sat, -degree[v], v))
    return color



if __name__ == '__main__':
    dir = '../../data/syn_BA'
    prefixes = ['BA_5_']
    files = calc_txt_files_with_prefixes(dir, prefixes)
    for i in range(len(files)):
        filename = files[i]
        graph = read_nxgraph(filename)
        result = degree_of_saturation(graph)
        print(result)
    pass
