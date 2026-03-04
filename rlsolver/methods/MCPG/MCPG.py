import os

cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../')

import yaml
import time
import tyro
from typing import List, Tuple
from base_solver import base_solver
from dataloader import dataloader_select2
from config import DEVICE, Problem, PROBLEM

from rlsolver.methods.util_read_data import read_set_cover_data, read_nxgraph
from rlsolver.methods.util_write_read_result import write_result_set_cover
from rlsolver.methods.util import (plot_fig,
                                   plot_nxgraph,
                                   transfer_nxgraph_to_weightmatrix,
                                   calc_txt_files_with_prefixes,
                                   )
from rlsolver.methods.util_write_read_result import write_graph_result
from dataloader import qubo_dataloader
from config import *

device = DEVICE

def run_one_file(problem: Problem, filename: str):
    if problem == PROBLEM.maxcut:
        config = ConfigMaxcut()
    elif problem == PROBLEM.maxcut_edge:
        config = ConfigMaxcutEdge()
    elif problem == PROBLEM.maxsat:
        config = ConfigMaxsat()
    elif problem == PROBLEM.MIMO:
        config = ConfigMIMO()
    elif problem == PROBLEM.ncheegercut:
        config = ConfigNcheegercut()
    elif problem == PROBLEM.partial_maxsat:
        config = ConfigPartialMaxsat()
    elif problem == PROBLEM.qubo_bin:
        config = ConfigQuboBin()
    elif problem == PROBLEM.qubo:
        config = ConfigQubo()
    elif problem == PROBLEM.rcheegercut:
        config = ConfigNcheegercut()
    else:
        raise ValueError("error, problem")
    # config = config.__dict__
    print("config: ", config)
    # config = class_to_dict_recursive(config)
    print("filename: ", filename)
    # args = LocalConfig()
    # args.config_file = config_file
    # args.problem_instance = problem_instance
    # with open(config_file) as f:
    #     config = yaml.safe_load(f)
    # path = rlsolver_path + problem_instance
    # path = args.problem_instance
    start_time = time.perf_counter()
    dataloader = dataloader_select2(problem)
    data, num_vars = dataloader(filename)
    dataloader_t = time.perf_counter()
    obj, solution, _, _ = base_solver(problem, num_vars, config, data, verbose=True)
    mcpg_t = time.perf_counter()

    if problem in [Problem.maxsat.value, Problem.partial_maxsat] and len(data.pdata) == 7:
        if obj > data.pdata[5] * data.pdata[6]:
            obj -= data.pdata[5] * data.pdata[6]
            print("SATISFIED")
            print("SATISFIED SOFT CLAUSES:", obj)
            print("UNSATISFIED SOFT CLAUSES:", data.pdata[1] - data.pdata[-1] - obj)
        else:
            obj = obj // data.pdata[5] - data.pdata[6]
            print("UNSATISFIED")
    else:
        print("OUTPUT: {:.2f}".format(obj))

    print("DATA LOADING TIME: {:.2f}".format(dataloader_t - start_time))
    print("MCPG RUNNING TIME: {:.2f}".format(mcpg_t - dataloader_t))
    return obj, solution


def run_manyfiles(alg_name, problem: Problem, data_dir: str, prefixes: List[str]) -> List[List[float]]:
    scores = []
    solutions = []
    files = calc_txt_files_with_prefixes(data_dir, prefixes)
    for i in range(len(files)):
        start_time = time.time()
        filename = files[i]
        print(f'Start the {i}-th file: {filename}')
        if problem in [Problem.qubo, Problem.qubo_bin, Problem.maxsat, Problem.partial_maxsat]:
            num_nodes = None
            plus1 = False
        else:
            graph = read_nxgraph(filename)
            num_nodes = graph.number_of_nodes()
            plus1 = True
        score, solution = run_one_file(PROBLEM, filename)
        scores.append(score)
        solutions.append(solution)
        running_duration = time.time() - start_time
        info_dict = {'problem': PROBLEM.value}
        write_graph_result(score, running_duration, num_nodes, alg_name, solution, filename, plus1=plus1, info_dict=info_dict)
    return scores, solutions


if __name__ == '__main__':
    PROBLEM = Problem.qubo_bin

    alg_name = "MCPG"
    if PROBLEM == Problem.maxcut:
        data_dir = rlsolver_path + "data/syn_BA"
        prefixes = ["BA_5_"]
    if PROBLEM == Problem.maxcut_edge:
        data_dir = rlsolver_path + "data/syn_BA"
        prefixes = ["BA_5_"]
    if PROBLEM == Problem.maxsat:
        data_dir = rlsolver_path + "data/sat"
        prefixes = ["randu0"]
    if PROBLEM == Problem.ncheegercut:
        data_dir = rlsolver_path + "data/syn_BA"
        prefixes = ["BA_5_"]
    if PROBLEM == Problem.partial_maxsat:
        data_dir = rlsolver_path + "data/partial_sat"
        prefixes = ["clq1-cv160c800l2g0"]
    if PROBLEM == Problem.qubo:
        data_dir = rlsolver_path + "data/nbiq"
        prefixes = ["nbiq_5"]
    if PROBLEM == Problem.qubo_bin:
        data_dir = rlsolver_path + "data/nbiq"
        prefixes = ["nbiq_5"]
    if PROBLEM == Problem.rcheegercut:
        data_dir = rlsolver_path + "data/syn_BA"
        prefixes = ["BA_5_"]

    # data_dir = rlsolver_path + "data/nbiq"
    # prefixes = ["nbiq_5"]
    scores, solutions = run_manyfiles(alg_name, PROBLEM, data_dir, prefixes)
    print()
