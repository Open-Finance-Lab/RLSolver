"""
Copyright (c) 2024 Cheng Chen, Ruitao Chen, Tianyou Li, Zaiwen Wen
All rights reserved.
Redistribution and use in source and binary forms, with or without modification, are permitted provided that
the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
   following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
   and the following disclaimer in the documentation and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import yaml
import time
import tyro
from mcpg_solver import mcpg_solver
from dataloader import dataloader_select
from config import DEVICE, Problem
device = DEVICE

class Config:
    config_file: str # "input the configuration file for the mcpg solver"
    problem_instance: str # "input the data file for the problem instance"


# parser = argparse.ArgumentParser()
# parser.add_argument("config_file", type=str,
#                     help="input the configuration file for the mcpg solver")
# parser.add_argument("problem_instance", type=str,
#                     help="input the data file for the problem instance")

# args = parser.parse_args()
def main(config_file: str, problem_instance: str):
    print("problem_instance: ", problem_instance)
    args = Config()
    args.config_file = config_file
    args.problem_instance = problem_instance
    with open(args.config_file) as f:
        config = yaml.safe_load(f)

    path = args.problem_instance
    start_time = time.perf_counter()
    dataloader = dataloader_select(config["problem_type"])
    data, nvar = dataloader(path)
    dataloader_t = time.perf_counter()
    res, solutions, _, _ = mcpg_solver(nvar, config, data, verbose=True)
    mcpg_t = time.perf_counter()

    if config["problem_type"] == Problem.maxsat.value and len(data.pdata) == 7:
        if res > data.pdata[5] * data.pdata[6]:
            res -= data.pdata[5] * data.pdata[6]
            print("SATISFIED")
            print("SATISFIED SOFT CLAUSES:", res)
            print("UNSATISFIED SOFT CLAUSES:", data.pdata[1] - data.pdata[-1] - res)
        else:
            res = res//data.pdata[5]-data.pdata[6]
            print("UNSATISFIED")

    elif "obj_type" in config and config["obj_type"] == "neg":
        print("exist neg in mcpy.py")
        print("OUTPUT: {:.2f}".format(-res))
    else:
        print("OUTPUT: {:.2f}".format(res))


    print("DATA LOADING TIME: {:.2f}".format(dataloader_t - start_time))
    print("MCPG RUNNING TIME: {:.2f}".format(mcpg_t - dataloader_t))

if __name__ == '__main__':
    tyro.cli(main)