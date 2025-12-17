import sys
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../..')
sys.path.append(os.path.dirname(rlsolver_path))


import time
from typing import List

from config import *

from rlsolver.methods.util_write_read_result import write_graph_result
from rlsolver.methods.util import calc_txt_files_with_prefixes


from rlsolver.methods.config import PROBLEM, Problem

PROBLEM = Problem.graph_coloring

def select_alg(input_alg):
    if input_alg == Alg.degree_of_saturation:
        from rlsolver.methods_problem_specific.graph_coloring.degree_of_saturation import degree_of_saturation as alg
        return alg
    elif input_alg == Alg.greedy:
        from rlsolver.methods_problem_specific.graph_coloring.greedy import greedy as alg
        return alg
    elif input_alg == Alg.recursive:
        from rlsolver.methods_problem_specific.graph_coloring.recursive import recursive as alg
        return alg
    elif input_alg == Alg.welsh_powell:
        from rlsolver.methods_problem_specific.graph_coloring.welsh_powell import welsh_powell as alg
        return alg
    else:
        raise ValueError("not supported graph coloring algorithm")

def run_over_manyfiles(alg, alg_name, directory_data: str, prefixes: List[str]) -> List[List[float]]:
    from rlsolver.methods.util_read_data import read_nxgraph
    scoress = []
    files = calc_txt_files_with_prefixes(directory_data, prefixes)
    for i in range(len(files)):
        start_time = time.time()
        filename = files[i]
        print("alg_name: ", alg_name)
        print(f'Start the {i}-th file: {filename}')
        graph = read_nxgraph(filename)
        alg2 = select_alg(alg)
        colors = alg2(graph)
        score = -len(set(colors))
        scoress.append(score)
        solution = colors
        running_duration = time.time() - start_time
        info_dict = {'problem': PROBLEM.value}
        write_graph_result(score, running_duration, graph.number_of_nodes(), alg_name, solution, filename, info_dict=info_dict)
    return scoress



def main():
    dir = '../../data/syn_BA'
    prefixes = ['BA_100_']
    ALG = Alg.greedy
    scoress = run_over_manyfiles(ALG, ALG.value, dir, prefixes)
    print("scoress: ", scoress)

    run_all_algs = True
    if run_all_algs:
        algs = list(Alg)
        for alg in algs:
            scoress = run_over_manyfiles(alg, alg.value, dir, prefixes)
            print("scoress: ", scoress)


    pass

if __name__ == '__main__':
    main()