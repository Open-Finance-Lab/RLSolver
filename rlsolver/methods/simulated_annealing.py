import sys
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

import copy
import time
from typing import Union, Optional
import numpy as np
import random
import networkx as nx

from rlsolver.methods.util import (calc_txt_files_with_prefixes,
                  plot_fig
                 )
from rlsolver.methods.util_read_data import (read_nxgraph,
                            read_set_cover_data, )
from rlsolver.methods.util_write_read_result import (write_graph_result,
                                                     write_result_set_cover
                                                     )
from rlsolver.methods.util_obj import (
    obj_maxcut,
    obj_graph_partitioning,
    cover_all_edges,
    obj_MVC,
    obj_MIS,
    obj_set_cover,
    obj_graph_coloring,
                      )
from rlsolver.methods.greedy import (greedy_maxcut,
                                     greedy_graph_partitioning,
                                     greedy_MVC,
                                     greedy_MIS,
                                     greedy_set_cover,
                                     greedy_graph_coloring,
                                     )
# from util import run_simulated_annealing_over_multiple_files
from rlsolver.methods.config import *


def simulated_annealing_set_cover(init_temperature: int,
                                  num_steps: Optional[int],
                                  num_items: int,
                                  num_sets: int,
                                  item_matrix: List[List[int]]) -> (int, Union[List[int], np.array], List[int]):
    print('simulated_annealing')
    start_time = time.time()
    gr_score, gr_solution, gr_scores = greedy_set_cover(num_items, num_sets, item_matrix)
    init_score = gr_score
    curr_solution = copy.deepcopy(gr_solution)
    curr_score = gr_score
    scores = []
    scores.append(init_score)
    for k in range(num_steps):
        # The temperature decreases
        temperature = init_temperature * (1 - (k + 1) / num_steps)
        new_solution = copy.deepcopy(curr_solution)
        selected_sets = []
        unselected_sets = []
        for i in range(len(new_solution)):
            if new_solution[i] == 1:
                selected_sets.append(i)
            else:
                unselected_sets.append(i)
        # if prob < prob_thresh, swap one set in selected_sets with one set in unselected_sets;
        # if prob > prob_thresh, swap two sets in selected_sets with one set in unselected_sets;
        prob_thresh = 0.05
        prob = random.random()
        idx_in = np.random.randint(0, len(unselected_sets))
        set_in = unselected_sets[idx_in]
        new_solution[set_in] = (new_solution[set_in] + 1) % 2
        if prob < prob_thresh:
            idx_out = np.random.randint(0, len(selected_sets))
            set_out = selected_sets[idx_out]
            new_solution[set_out] = (new_solution[set_out] + 1) % 2
        else:
            while True:
                idx_out1, idx_out2 = np.random.randint(0, len(selected_sets), 2)
                if idx_out1 != idx_out2:
                    break
            set_out1 = selected_sets[idx_out1]
            set_out2 = selected_sets[idx_out2]
            new_solution[set_out1] = (new_solution[set_out1] + 1) % 2
            new_solution[set_out2] = (new_solution[set_out2] + 1) % 2
        new_score = obj_set_cover(new_solution, num_items, item_matrix)
        store = False
        delta_e = curr_score - new_score
        if delta_e < 0:
            curr_solution = new_solution
            curr_score = new_score
            store = True
        else:
            prob = np.exp(-delta_e / (temperature + 1e-6))
            if prob > random.random():
                curr_solution = new_solution
                curr_score = new_score
                store = True
        if store:
            scores.append(new_score)
    print("init_score, final score of simulated_annealing", init_score, curr_score)
    print("scores: ", scores)
    print("solution: ", curr_solution)
    running_duration = time.time() - start_time
    print('running_duration: ', running_duration)
    return curr_score, curr_solution, scores


def simulated_annealing(init_temperature: int, num_steps: Optional[int], graph: nx.Graph, filename) \
        -> (int, Union[List[int], np.array], List[int]):
    print('simulated_annealing')
    num_nodes = int(graph.number_of_nodes())
    if PROBLEM == Problem.maxcut:
        if num_steps is None:
            num_steps = num_nodes
        gr_score, gr_solution, gr_scores = greedy_maxcut(num_steps, graph, filename)
    elif PROBLEM == Problem.graph_partitioning:
        num_steps = num_nodes
        gr_score, gr_solution, gr_scores = greedy_graph_partitioning(num_steps, graph)
    elif PROBLEM == Problem.MVC:
        num_steps = num_nodes
        gr_score, gr_solution, gr_scores = greedy_MVC(None, graph)
        assert cover_all_edges(gr_solution, graph)
    elif PROBLEM == Problem.MIS:
        num_steps = 100 * num_nodes
        gr_score, gr_solution, gr_scores = greedy_MIS(num_steps, graph)
    elif PROBLEM == Problem.graph_coloring:
        num_steps = None
        gr_score, gr_solution, gr_scores = greedy_graph_coloring(num_steps, graph)
        num_steps = 10 * num_nodes

    start_time = time.time()
    init_score = gr_score
    curr_solution = copy.deepcopy(gr_solution)
    curr_score = gr_score
    scores = []
    scores.append(init_score)

    for k in range(num_steps):
        # The temperature decreases
        temperature = init_temperature * (1 - (k + 1) / num_steps)
        new_solution = copy.deepcopy(curr_solution)
        if PROBLEM == Problem.maxcut:
            idx = np.random.randint(0, num_nodes)
            new_solution[idx] = (new_solution[idx] + 1) % 2
            new_score = obj_maxcut(new_solution, graph)
        elif PROBLEM == Problem.graph_partitioning:
            while True:
                idx = np.random.randint(0, num_nodes)
                node2 = np.random.randint(0, num_nodes)
                if new_solution[idx] != new_solution[node2]:
                    break
            print(f"new_solution[index]: {new_solution[idx]}, new_solution[index2]: {new_solution[node2]}")
            tmp = new_solution[idx]
            new_solution[idx] = new_solution[node2]
            new_solution[node2] = tmp
            new_score = obj_graph_partitioning(new_solution, graph)
        elif PROBLEM == Problem.MVC:
            iter = 0
            index = None
            while True:
                iter += 1
                if iter >= num_steps:
                    break
                indices_eq_1 = []
                for i in range(len(new_solution)):
                    if new_solution[i] == 1:
                        indices_eq_1.append(i)
                idx = np.random.randint(0, len(indices_eq_1))
                new_solution2 = copy.deepcopy(new_solution)
                new_solution2[indices_eq_1[idx]] = 0
                if cover_all_edges(new_solution2, graph):
                    index = indices_eq_1[idx]
                    break
            if index is not None:
                new_solution[index] = 0
            new_score = obj_MVC(new_solution, graph, False)
        elif PROBLEM == Problem.MIS:
            selected_indices = []
            unselected_indices = []
            for i in range(len(new_solution)):
                if new_solution[i] == 1:
                    selected_indices.append(i)
                else:
                    unselected_indices.append(i)
            idx_out = np.random.randint(0, len(selected_indices))
            node_out = selected_indices[idx_out]
            new_solution[node_out] = (new_solution[node_out] + 1) % 2
            # if prob < prob_thresh, change one node; if prob > prob_thresh, change two nodes
            prob_thresh = 0.05
            prob = random.random()
            if prob < prob_thresh:
                idx_in = np.random.randint(0, len(unselected_indices))
                node_in = unselected_indices[idx_in]
                new_solution[node_in] = (new_solution[node_in] + 1) % 2
            else:
                while True:
                    node1, node2 = np.random.randint(0, len(unselected_indices), 2)
                    if node1 != node2:
                        break
                node_in1 = unselected_indices[node1]
                node_in2 = unselected_indices[node2]
                new_solution[node_in1] = (new_solution[node_in1] + 1) % 2
                new_solution[node_in2] = (new_solution[node_in2] + 1) % 2
            new_score = obj_MIS(new_solution, graph)
        elif PROBLEM == Problem.graph_coloring:
            while True:
                node1, node2 = np.random.randint(0, num_nodes, 2)
                if node1 != node2:
                    break
            tmp_color = new_solution[node1]
            new_solution[node1] = new_solution[node2]
            new_solution[node2] = tmp_color
            new_score = obj_graph_coloring(new_solution, graph)
        store = False
        delta_e = curr_score - new_score
        if delta_e < 0:
            curr_solution = new_solution
            curr_score = new_score
            store = True
        else:
            prob = np.exp(-delta_e / (temperature + 1e-6))
            if prob > random.random():
                curr_solution = new_solution
                curr_score = new_score
                store = True
        if store:
            scores.append(new_score)
    print("init_score, final score of simulated_annealing", init_score, curr_score)
    print("scores: ", scores)
    print("solution: ", curr_solution)
    running_duration = time.time() - start_time
    print('running_duration: ', running_duration)
    return curr_score, curr_solution, scores

def run_simulated_annealing_over_multiple_files(alg, alg_name, init_temperature, num_steps, directory_data: str, prefixes: List[str])-> List[List[float]]:
    scoress = []
    files = calc_txt_files_with_prefixes(directory_data, prefixes)
    for i in range(len(files)):
        start_time = time.time()
        filename = files[i]
        print(f'Start the {i}-th file: {filename}')
        if PROBLEM == Problem.set_cover:
            num_items, num_sets, item_matrix = read_set_cover_data(filename)
            init_temperature = 4
            num_steps = int(50 * num_sets)
            score, solution, scores = simulated_annealing_set_cover(init_temperature, num_steps,
                                                                    num_items, num_sets, item_matrix)
            scoress.append(scores)
            running_duration = time.time() - start_time
            alg_name = 'greedy'
            write_result_set_cover(score, running_duration, num_items, num_sets, alg_name, filename)
        else:
            graph = read_nxgraph(filename)
            score, solution, scores = alg(init_temperature, num_steps, graph, filename)
            scoress.append(scores)
            running_duration = time.time() - start_time
            num_nodes = int(graph.number_of_nodes())
            write_graph_result(score, running_duration, num_nodes, alg_name, solution, filename)
    return scoress


if __name__ == '__main__':
    print(f'problem: {PROBLEM}')
    alg = simulated_annealing
    alg_name = 'simulated_annealing'
    if_run_graph_based_problems = True
    if if_run_graph_based_problems:
        init_temperature = 0.2
        num_steps = None
        directory_data = '../data/syn_BA'
        prefixes = ['barabasi_albert_100_']
    else:
        init_temperature = 4
        num_steps = None
        directory_data = '../data/set_cover'
        prefixes = ['frb30-15-1.msc']
    run_simulated_annealing_over_multiple_files(alg, alg_name, init_temperature, num_steps, directory_data, prefixes)
