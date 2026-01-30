
import torch
import time
import yaml
import argparse
from dataloader import read_data_mimo
from MCPG_solver import MCPG_solver
import tyro
from config import ConfigMIMO, Problem, PROBLEM

class ConfigLocalMimo:
    snr: int = 2
    size: int = 180

def main():
    problem: Problem = Problem.MIMO
    args = ConfigLocalMimo()
    # args.snr = snr
    # args.size = size

    H_num = 10
    X_num = 10
    num_rand = H_num * X_num

    # parser = argparse.ArgumentParser()
    # parser.add_argument("snr", type=int,
    #                     help="input the SNR for the simulation.")
    # parser.add_argument("size", type=int,
    #                     help="input the problem size which appoint the dataset.")
    # args = parser.parse_args()
    # with open("config/mimo_default.yaml") as f:
    #     config = yaml.safe_load(f)
    config = ConfigMIMO()
    print("args: ", args)
    print("args.snr: ", args.snr)
    print("args.size: ", args.size)

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

    num_nodes = 2*args.size
    max_range = 150
    config.num_ls = 4  # local search epoch
    config, max_range = get_parameter(args.snr, args.size, args.size, config)  # get paramter

    rand_seeds = list(range(0, num_rand))

    record = []
    total_time = 0

    for r_seed in rand_seeds:

        data = read_data_mimo(args.size, args.size, args.snr, X_num, r_seed)

        total_start_time = time.perf_counter()
        _, _, now_best, now_best_info = MCPG_solver(problem, num_nodes, config, data)
        now_best_info = now_best_info * 2 - 1
        total_end_time = time.perf_counter()
        total_time += total_end_time - total_start_time
        # get average information
        best_sort = torch.argsort(now_best, descending=True)
        total_best_info = torch.squeeze(now_best_info[:, best_sort[0]])
        for i0 in range(max_range):
            total_best_info += torch.squeeze(now_best_info[:, best_sort[i0]])
        total_best_info = torch.sign(total_best_info)
        record.append((total_best_info != data[2]).sum().item()/num_nodes)

    print("SNR = {} SIZE = {}".format(args.snr, args.size))
    print("BEST: ", min(record))
    print("MEAN: ", sum(record)/num_rand)
    print("AVERAGE TIME: ", total_time/num_rand)

if __name__ == '__main__':
    # tyro.cli(main)
    main()