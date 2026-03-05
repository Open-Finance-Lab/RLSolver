import copy
import os

cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../')

import yaml
import time
import tyro
from typing import List, Tuple
from base_solver import base_solver
from dataloader import dataloader_select
from rlsolver.methods.MCPG.config import DEVICE, Problem, PROBLEM
from rlsolver.methods.MCPG.config import ConfigMaxcut, ConfigMaxcutEdge, ConfigMaxsat, ConfigMIMO, ConfigLocalMimo, ConfigNcheegercut, ConfigPartialMaxsat, ConfigQubo, ConfigQuboBin

from rlsolver.methods.util_read_data import read_nxgraph
from rlsolver.methods.util import calc_txt_files_with_prefixes

from rlsolver.methods.util_write_read_result import write_graph_result

import torch
import time
from dataloader import read_data_mimo5


device = DEVICE

def run_one_file(problem: Problem, filename: str):
    if problem == Problem.maxcut:
        config = ConfigMaxcut()
    elif problem == Problem.maxcut_edge:
        config = ConfigMaxcutEdge()
    elif problem == Problem.maxsat:
        config = ConfigMaxsat()
    elif problem == Problem.MIMO:
        config = ConfigMIMO()
    elif problem == Problem.ncheegercut:
        config = ConfigNcheegercut()
    elif problem == Problem.partial_maxsat:
        config = ConfigPartialMaxsat()
    elif problem == Problem.qubo_bin:
        config = ConfigQuboBin()
    elif problem == Problem.qubo:
        config = ConfigQubo()
    elif problem == Problem.rcheegercut:
        config = ConfigNcheegercut()
    else:
        raise ValueError("error, problem")
    print("config: ", config)
    print("filename: ", filename)
    start_time = time.perf_counter()
    dataloader = dataloader_select(problem)
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
        score, solution = run_one_file(problem, filename)
        scores.append(score)
        solutions.append(solution)
        running_duration = time.time() - start_time
        info_dict = {'problem': problem.value}
        write_graph_result(score, running_duration, num_nodes, alg_name, solution, filename, plus1=plus1, info_dict=info_dict)
    return scores, solutions


def MCPG_MIMO(config_local_mimo: ConfigLocalMimo, config: ConfigMIMO):
    H_num = 10
    X_num = 10
    num_rand = H_num * X_num

    print("config_local_mimo: ", config_local_mimo)
    print("config_local_mimo.snr: ", config_local_mimo.snr)
    print("config_local_mimo.size: ", config_local_mimo.size)

    def get_parameter(SNR, N, K, config):
        config.num_ls = 4
        num_epochs = 3
        max_range = 150
        if N == 800 and K == 800:
            if SNR == 2:
                config.num_ls = 3
                num_epochs = 1
            elif SNR == 4:
                config.num_ls = 3
                num_epochs = 2
            elif SNR == 6:
                config.num_ls = 5
                num_epochs = 5
            elif SNR == 8:
                config.num_ls = 7
                num_epochs = 3
            elif SNR == 10:
                config.num_ls = 5
                num_epochs = 2
            elif SNR == 12:
                config.num_ls = 4
                num_epochs = 2
        if N == 1000 and K == 1000:
            if SNR == 2:
                config.num_ls = 2
                num_epochs = 2
            elif SNR == 4:
                config.num_ls = 3
                num_epochs = 2
            elif SNR == 6:
                config.num_ls = 7
                num_epochs = 4
            elif SNR == 8:
                config.num_ls = 8
                num_epochs = 4
            elif SNR == 10:
                config.num_ls = 7
                num_epochs = 2
            elif SNR == 12:
                config.num_ls = 3
                num_epochs = 3
        if N == 1200 and K == 1200:
            if SNR == 2:
                config.num_ls = 3
                num_epochs = 1
            elif SNR == 4:
                config.num_ls = 3
                num_epochs = 2
            elif SNR == 6:
                config.num_ls = 7
                num_epochs = 3
            elif SNR == 8:
                config.num_ls = 6
                num_epochs = 4
                max_range = 100
            elif SNR == 10:
                config.num_ls = 4
                num_epochs = 4
            elif SNR == 12:
                config.num_ls = 5
                num_epochs = 2
        config.max_epoch_num = num_epochs * config.sample_epoch_num
        return config, max_range

    num_nodes = 2 * config_local_mimo.size
    config.num_ls = 4  # local search epoch
    config, max_range = get_parameter(config_local_mimo.snr, config_local_mimo.size, config_local_mimo.size, config)  # get paramter

    record = []
    total_time = 0
    problem = Problem.MIMO
    for r_seed in range(num_rand):
        data = read_data_mimo5(config_local_mimo.size, config_local_mimo.size, config_local_mimo.snr, X_num, r_seed)
        total_start_time = time.perf_counter()
        _, _, now_best, now_best_info = base_solver(problem, num_nodes, config, data)
        now_best_info = now_best_info * 2 - 1
        total_end_time = time.perf_counter()
        total_time += total_end_time - total_start_time
        # get average information
        best_sort = torch.argsort(now_best, descending=True)
        total_best_info = torch.squeeze(now_best_info[:, best_sort[0]])
        for i0 in range(max_range):
            total_best_info += torch.squeeze(now_best_info[:, best_sort[i0]])
        total_best_info = torch.sign(total_best_info)
        record.append((total_best_info != data[2]).sum().item() / num_nodes)

    print("SNR = {} SIZE = {}".format(config_local_mimo.snr, config_local_mimo.size))
    print("BEST: ", min(record))
    print("MEAN: ", sum(record) / num_rand)
    print("AVERAGE TIME: ", total_time / num_rand)

if __name__ == '__main__':
    PROBLEM = Problem.maxcut

    alg_name = "MCPG"
    if PROBLEM == Problem.maxcut:
        data_dir = rlsolver_path + "data/syn_BA"
        prefixes = ["BA_5_"]
    if PROBLEM == Problem.maxcut_edge:
        data_dir = rlsolver_path + "data/syn_BA"
        prefixes = ["BA_5_"]
    if PROBLEM == Problem.maxsat:
        data_dir = rlsolver_path + "data/maxsat"
        prefixes = ["randu0"]
    if PROBLEM == Problem.ncheegercut:
        data_dir = rlsolver_path + "data/syn_BA"
        prefixes = ["BA_5_"]
    if PROBLEM == Problem.partial_maxsat:
        data_dir = rlsolver_path + "data/partial_maxsat"
        prefixes = ["clq1-cv160c800l2g0"]
    if PROBLEM == Problem.qubo:
        data_dir = rlsolver_path + "data/qubo"
        prefixes = ["nbiq_5"]
    if PROBLEM == Problem.qubo_bin:
        data_dir = rlsolver_path + "data/qubo"
        prefixes = ["nbiq_5"]
    if PROBLEM == Problem.rcheegercut:
        data_dir = rlsolver_path + "data/syn_BA"
        prefixes = ["BA_5_"]

    # data_dir = rlsolver_path + "data/qubo"
    # prefixes = ["nbiq_5"]
    if PROBLEM == Problem.MIMO:
        config_local_mimo = ConfigLocalMimo()
        config = ConfigMIMO()
        MCPG_MIMO(config_local_mimo, config)
    else:
        scores, solutions = run_manyfiles(alg_name, PROBLEM, data_dir, prefixes)
    print()
