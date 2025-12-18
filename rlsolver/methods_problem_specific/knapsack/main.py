import sys
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../..')
sys.path.append(os.path.dirname(rlsolver_path))


import time
from typing import List

from config import *

from rlsolver.methods.util_write_read_result import write_result_knapsack
from rlsolver.methods.util import calc_txt_files_with_prefixes
from rlsolver.methods.util_read_data import read_knapsack_data


from rlsolver.methods.config import PROBLEM, Problem

PROBLEM = Problem.knapsack

def select_alg(input_alg):
    if input_alg == Alg.branch_bound:
        from rlsolver.methods_problem_specific.knapsack.branch_bound import branch_and_bound as alg
        return alg
    if input_alg == Alg.brute_force:
        from rlsolver.methods_problem_specific.knapsack.brute_force import brute_force as alg
        return alg
    if input_alg == Alg.dynamic_programming:
        from rlsolver.methods_problem_specific.knapsack.dynamic_programming import dynamic_programming as alg
        return alg
    elif input_alg == Alg.fptas:
        from rlsolver.methods_problem_specific.knapsack.fptas import fptas as alg
        return alg
    elif input_alg == Alg.greedy:
        from rlsolver.methods_problem_specific.knapsack.greedy import greedy as alg
        return alg
    elif input_alg == Alg.simulated_annealing:
        from rlsolver.methods_problem_specific.knapsack.simulated_annealing import simulated_annealing as alg
        return alg
    else:
        raise ValueError("not supported: ", input_alg)

def run_over_manyfiles(alg, alg_name, directory_data: str, prefixes: List[str]) -> List[List[float]]:
    from rlsolver.methods.util_read_data import read_nxgraph
    scoress = []
    files = calc_txt_files_with_prefixes(directory_data, prefixes)
    for i in range(len(files)):
        start_time = time.time()
        filename = files[i]
        print("alg_name: ", alg_name)
        print(f'Start the {i}-th file: {filename}')
        instance_id, num_items, capacity, weights, profits = read_knapsack_data(filename)
        alg2 = select_alg(alg)
        weights_profits = []
        for i in range(len(weights)):
            weights_profits.append((weights[i], profits[i]))
        score, best_combination = alg2(num_items, capacity, weights_profits)
        scoress.append(score)
        solution = best_combination
        running_duration = time.time() - start_time
        info_dict = {'problem': PROBLEM.value}
        write_result_knapsack(score, running_duration, num_items, alg_name, solution, filename, info_dict=info_dict)
    return scoress



def main():
    run_one_alg = True
    if run_one_alg:
        dir = '../../data/knapsack'
        prefixes = ['knap_4_']
        ALG = Alg.greedy
        scoress = run_over_manyfiles(ALG, ALG.value, dir, prefixes)
        print("scoress: ", scoress)

    run_all_algs = False
    if run_all_algs:
        algs = list(Alg)
        for alg in algs:
            scoress = run_over_manyfiles(alg, alg.value, dir, prefixes)
            print("scoress: ", scoress)


    pass

if __name__ == '__main__':
    main()