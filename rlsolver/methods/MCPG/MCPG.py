import os

cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../')

import yaml
import time
import tyro
from typing import List, Tuple
from MCPG_solver import mcpg_solver
from dataloader import dataloader_select
from config import DEVICE, Problem, PROBLEM

from rlsolver.methods.util_read_data import (read_set_cover_data, read_nxgraph)
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


def run_old(config, problem_instance: str):
    print("problem_instance: ", problem_instance)

    # with open(config_file) as f:
    #     config = yaml.safe_load(f)

    path = rlsolver_path + problem_instance
    start_time = time.perf_counter()
    dataloader = dataloader_select(config.problem_type)
    data, num_vars = dataloader(path)
    # device2 = data['Q'].device
    # print("device2: ", device2)
    dataloader_t = time.perf_counter()
    res, solution, _, _ = mcpg_solver(num_vars, config, data, verbose=True)
    mcpg_t = time.perf_counter()

    if config.problem_type == Problem.maxsat.value and len(data.pdata) == 7:
        if res > data.pdata[5] * data.pdata[6]:
            res -= data.pdata[5] * data.pdata[6]
            print("SATISFIED")
            print("SATISFIED SOFT CLAUSES:", res)
            print("UNSATISFIED SOFT CLAUSES:", data.pdata[1] - data.pdata[-1] - res)
        else:
            res = res//data.pdata[5]-data.pdata[6]
            print("UNSATISFIED")
    else:
        print("OUTPUT: {:.2f}".format(res))

    print("DATA LOADING TIME: {:.2f}".format(dataloader_t - start_time))
    print("MCPG RUNNING TIME: {:.2f}".format(mcpg_t - dataloader_t))

def run_one_file(problem: Problem, filename: str):
    if problem == PROBLEM.maxcut:
        config = ConfigMaxcut()
    elif problem == PROBLEM.maxcut_edge:
        config = ConfigMaxcutEdge()
    elif problem == PROBLEM.maxsat:
        config = ConfigMaxsat()
    elif problem == PROBLEM.MIMO:
        config = ConfigMIMO()
    elif problem == PROBLEM.n_cheegercut:
        config = ConfigNcheegercut()
    elif problem == PROBLEM.partial_maxsat:
        config = ConfigPartialMaxsat()
    elif problem == PROBLEM.qubo_bin:
        config = ConfigQuboBin()
    elif problem == PROBLEM.qubo:
        config = ConfigQubo()
    elif problem == PROBLEM.r_cheegercut:
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
    dataloader = dataloader_select(config.problem_type)
    data, num_vars = dataloader(filename)
    dataloader_t = time.perf_counter()
    obj, solution, _, _ = mcpg_solver(num_vars, config, data, verbose=True)
    mcpg_t = time.perf_counter()

    if config.problem_type == Problem.maxsat.value and len(data.pdata) == 7:
        if obj > data.pdata[5] * data.pdata[6]:
            obj -= data.pdata[5] * data.pdata[6]
            print("SATISFIED")
            print("SATISFIED SOFT CLAUSES:", obj)
            print("UNSATISFIED SOFT CLAUSES:", data.pdata[1] - data.pdata[-1] - obj)
        else:
            obj = obj//data.pdata[5]-data.pdata[6]
            print("UNSATISFIED")
    else:
        print("OUTPUT: {:.2f}".format(obj))


    print("DATA LOADING TIME: {:.2f}".format(dataloader_t - start_time))
    print("MCPG RUNNING TIME: {:.2f}".format(mcpg_t - dataloader_t))
    return obj, solution


def run_manyfiles(alg_name, problem: Problem, data_dir: str, prefixes: List[str])-> List[List[float]]:
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
    # tyro.cli(main)
    run_maxcut = True
    run_qubo = True
    run_qubo_bin = True
    run_rcheegercut = True
    run_ncheegercut = True
    run_maxsat = True
    run_partial_sat = True

    run_old_version: bool = False
    if run_old_version:
        if run_maxcut:
            PROBLEM = Problem.maxcut
            problem_instance = "data/syn_BA/BA_5_ID0.txt"
        if run_qubo:
            PROBLEM = Problem.qubo
            problem_instance = "data/nbiq/nbiq_5.txt"
        if run_qubo_bin:
            PROBLEM = Problem.qubo_bin
            problem_instance = "data/nbiq/nbiq_5.txt"
        if run_rcheegercut:
            PROBLEM = Problem.r_cheegercut
            problem_instance = "data/syn_BA/BA_5_ID0.txt"
        if run_ncheegercut:
            PROBLEM = Problem.n_cheegercut
            problem_instance = "data/syn_BA/BA_5_ID0.txt"
        if run_maxsat:
            PROBLEM = Problem.maxsat
            problem_instance = "data/sat/randu0.cnf"
        if run_partial_sat:
            PROBLEM = Problem.partial_maxsat
            problem_instance = "data/partial_sat/clq1-cv160c800l2g0.wcnf"
        if PROBLEM == PROBLEM.maxcut:
            config = ConfigMaxcut()
        elif PROBLEM == PROBLEM.maxcut_edge:
            config = ConfigMaxcutEdge()
        elif PROBLEM == PROBLEM.maxsat:
            config = ConfigMaxsat()
        elif PROBLEM == PROBLEM.MIMO:
            config = ConfigMIMO()
        elif PROBLEM == PROBLEM.n_cheegercut:
            config = ConfigNcheegercut()
        elif PROBLEM == PROBLEM.partial_maxsat:
            config = ConfigPartialMaxsat()
        elif PROBLEM == PROBLEM.qubo_bin:
            config = ConfigQuboBin()
        elif PROBLEM == PROBLEM.qubo:
            config = ConfigQubo()
        elif PROBLEM == PROBLEM.r_cheegercut:
            config = ConfigNcheegercut()
        else:
            raise ValueError("error, problem")

        run_old(config, problem_instance)


    run_new_version: bool = True
    if run_new_version:
        alg_name = "MCPG"
        if run_maxcut:
            PROBLEM = Problem.maxcut
            data_dir = rlsolver_path + "data/syn_BA"
            prefixes = ["BA_5_"]
        if run_qubo:
            PROBLEM = Problem.qubo
            data_dir = rlsolver_path + "data/nbiq"
            prefixes = ["nbiq_5"]
        if run_qubo_bin:
            PROBLEM = Problem.qubo_bin
            data_dir = rlsolver_path + "data/nbiq"
            prefixes = ["nbiq_5"]
        if run_rcheegercut:
            PROBLEM = Problem.r_cheegercut
            data_dir = rlsolver_path + "data/syn_BA"
            prefixes = ["BA_5_"]
        if run_ncheegercut:
            PROBLEM = Problem.n_cheegercut
            data_dir = rlsolver_path + "data/syn_BA"
            prefixes = ["BA_5_"]
        if run_maxsat:
            PROBLEM = Problem.maxsat
            data_dir = rlsolver_path + "data/sat"
            prefixes = ["randu0"]
        if run_partial_sat:
            PROBLEM = Problem.partial_maxsat
            data_dir = rlsolver_path + "data/partial_sat"
            prefixes = ["clq1-cv160c800l2g0"]

        # data_dir = rlsolver_path + "data/nbiq"
        # prefixes = ["nbiq_5"]
        scores, solutions = run_manyfiles(alg_name, PROBLEM, data_dir, prefixes)
    print()
