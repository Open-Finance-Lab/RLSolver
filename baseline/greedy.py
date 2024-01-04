# comparison methods for maxcut: random walk, greedy, epsilon greedy, simulated annealing
import copy
import time
from typing import List, Union
import numpy as np
from typing import List
import networkx as nx
import itertools
from util import read_nxgraph
from util import obj_maxcut, obj_graph_partitioning, obj_minimum_vertex_cover
from util import write_result
from util import plot_fig
from util import transfer_nxgraph_to_weightmatrix
from util import cover_all_edges
from config import *

def greedy_maxcut(init_solution: Union[List[int], np.array], num_steps: int, graph: nx.Graph) -> (int, Union[List[int], np.array], List[int]):
    print('greedy')
    start_time = time.time()
    num_nodes = len(init_solution)
    nodes = list(range(num_nodes))
    curr_solution = copy.deepcopy(init_solution)
    curr_score: int = obj_maxcut(curr_solution, graph)
    init_score = curr_score
    scores = []
    for iteration in range(num_nodes):
        if iteration >= num_steps:
            break
        print("iteration in greedy: ", iteration)
        traversal_scores = []
        traversal_solutions = []
        # calc the new solution when moving to a new node. Then store the scores and solutions.
        for node in nodes:
            new_solution = copy.deepcopy(curr_solution)
            # search a new solution and calc obj
            new_solution[node] = (new_solution[node] + 1) % 2
            new_score = obj_maxcut(new_solution, graph)
            traversal_scores.append(new_score)
            traversal_solutions.append(new_solution)
        best_score = max(traversal_scores)
        index = traversal_scores.index(best_score)
        best_solution = traversal_solutions[index]
        if best_score >= curr_score:
            scores.append(best_score)
            curr_score = best_score
            curr_solution = best_solution
    print("score, init_score of greedy", curr_score, init_score)
    print("scores: ", traversal_scores)
    print("solution: ", curr_solution)
    running_duration = time.time() - start_time
    print('running_duration: ', running_duration)
    return curr_score, curr_solution, scores

def greedy_graph_partitioning(init_solution: Union[List[int], np.array], graph: nx.Graph) -> (int, Union[List[int], np.array], List[int]):
    print('greedy')
    start_time = time.time()
    num_nodes = len(init_solution)
    nodes = list(range(num_nodes))
    curr_solution = copy.deepcopy(init_solution)
    curr_score: int = obj_graph_partitioning(curr_solution, graph)
    init_score = curr_score
    scores = []
    for i in range(num_nodes):
        node1 = nodes[i]
        traversal_scores = []
        traversal_solutions = []
        for j in range(i + 1, num_nodes):
            node2 = nodes[j]
            new_solution = copy.deepcopy(curr_solution)
            tmp = new_solution[node1]
            new_solution[node1] = new_solution[node2]
            new_solution[node2] = tmp
            new_score = obj_graph_partitioning(new_solution, graph)
            traversal_scores.append(new_score)
            traversal_solutions.append(new_solution)
        if len(traversal_scores) == 0:
            continue
        best_score = max(traversal_scores)
        index = traversal_scores.index(best_score)
        best_solution = traversal_solutions[index]
        if best_score >= curr_score:
            scores.append(best_score)
            curr_score = best_score
            curr_solution = best_solution
    print("score, init_score of greedy", curr_score, init_score)
    print("scores: ", traversal_scores)
    print("solution: ", curr_solution)
    running_duration = time.time() - start_time
    print('running_duration: ', running_duration)
    return curr_score, curr_solution, scores



def greedy_weak_minimum_vertex_cover(init_solution: Union[List[int], np.array], graph: nx.Graph) -> (int, Union[List[int], np.array], List[int]):
    print('greedy')
    assert sum(init_solution) == 0
    start_time = time.time()
    num_nodes = len(init_solution)
    nodes = list(range(num_nodes))
    curr_solution = copy.deepcopy(init_solution)
    curr_score: int = obj_minimum_vertex_cover(curr_solution, graph)
    init_score = curr_score
    scores = []
    visited = [0] * num_nodes
    uncovered_edges = list(graph.edges)
    while True:
        cover_all = cover_all_edges(curr_solution, graph)
        if cover_all:
            break
        if sum(visited) == num_nodes:
            break
        index = np.random.randint(0, len(uncovered_edges))
        node1, node2 = uncovered_edges[index]
        curr_solution[node1] = 1
        curr_solution[node2] = 1
        visited[node1] = 1
        visited[node2] = 1
        uncovered_edges.pop(index)
    curr_score = obj_minimum_vertex_cover(curr_solution, graph)
    print("score, init_score of greedy", curr_score, init_score)
    print("solution: ", curr_solution)
    running_duration = time.time() - start_time
    print('running_duration: ', running_duration)
    return curr_score, curr_solution, scores

def greedy_strong_minimum_vertex_cover(init_solution: Union[List[int], np.array], graph: nx.Graph) -> (int, Union[List[int], np.array], List[int]):
    print('greedy')
    start_time = time.time()
    num_nodes = len(init_solution)
    nodes = list(range(num_nodes))
    curr_solution = copy.deepcopy(init_solution)
    curr_score: int = obj_minimum_vertex_cover(curr_solution, graph)
    init_score = curr_score
    scores = []
    visited = [0] * num_nodes
    while True:
        cover_all = cover_all_edges(curr_solution, graph)
        if cover_all:
            break
        if sum(visited) == num_nodes:
            break
        max_degree = 0
        best_node = -INF
        for i in range(num_nodes):
            node = nodes[i]
            degree = graph.degree(node)
            if visited[node] == 0 and degree > max_degree:
                max_degree = degree
                best_node = node
        if max_degree > 0:
            curr_solution[best_node] = 1
            visited[best_node] = 1
    if not cover_all_edges(curr_solution, graph):
        curr_score = -INF
    print("score, init_score of greedy", curr_score, init_score)
    print("solution: ", curr_solution)
    running_duration = time.time() - start_time
    print('running_duration: ', running_duration)
    return curr_score, curr_solution, scores

if __name__ == '__main__':
    # read data
    graph = read_nxgraph('../data/syn/syn_50_176.txt')
    weightmatrix = transfer_nxgraph_to_weightmatrix(graph)
    # run alg
    init_solution = [0] * int(graph.number_of_nodes() / 2) + [1] * int(graph.number_of_nodes() / 2)
    num_steps = 30
    alg_name = 'GR'

    # maxcut
    if PROBLEM == Problem.maxcut:
        gr_score, gr_solution, gr_scores = greedy_maxcut(init_solution, num_steps, graph)

    # graph_partitioning
    if PROBLEM == Problem.graph_partitioning:
        gr_score, gr_solution, gr_scores = greedy_graph_partitioning(init_solution, graph)

    if PROBLEM == Problem.minimum_vertex_cover:
        init_solution = [0] * graph.number_of_nodes()
        gr_score, gr_solution, gr_scores = greedy_weak_minimum_vertex_cover(init_solution, graph)
        obj = obj_minimum_vertex_cover(gr_solution, graph)
        print('obj: ', obj)

        gr_score, gr_solution, gr_scores = greedy_strong_minimum_vertex_cover(init_solution, graph)
        obj = obj_minimum_vertex_cover(gr_solution, graph)
        print('obj: ', obj)

    # write result
    write_result(gr_solution, '../result/result.txt')

    
    # plot fig
    plot_fig(gr_scores, alg_name)




