import os.path

import torch as th
import torch.nn as nn
from copy import deepcopy
import numpy as np
from torch import Tensor
from typing import List, Union, Tuple

from rlsolver.envs.env_k_spin_Ising import EnvMaxcut

from net import OptNet
import pickle as pkl
from rlsolver.methods.util_read_data import read_nxgraph
from rlsolver.methods.util_write_read_result import calc_result_file_name

# write a tensor/list/np.array (dim: 1) to a txt file.
# The nodes start from 0, and the label of classified set is 0 or 1 in our codes, but the nodes written to file start from 1, and the label is 1 or 2
def local_write_result(result: Union[Tensor, List, np.array],
                       filename: str = './result/result.txt',
                       obj: Union[int, float] = None,
                       running_duration: Union[int, float] = None):
    # assert len(result.shape) == 1
    # N = result.shape[0]
    num_nodes = len(result)
    directory = filename.split('/')[0]
    if not os.path.exists(directory):
        os.mkdir(directory)
    with open(filename, 'w', encoding="UTF-8") as file:
        if obj is not None:
            file.write(f'// obj: {obj}\n')
        if running_duration is not None:
            file.write(f'// running_duration: {running_duration}\n')
        for node in range(num_nodes):
            file.write(f'{node + 1} {int(result[node] + 1)}\n')

def train(
          filename: str,
          num_nodes: int,
          num_envs: int,
          device: th.device,
          opt_net: OptNet,
          optimizer: th.optim,
          episode_length: int,
          hidden_layer_size: int):
    env_maxcut = EnvMaxcut(filename=filename, num_samples=num_envs, device=device, episode_length=episode_length)

    num_layers = 1
    init_hidden = th.zeros(num_layers, num_envs, hidden_layer_size).to(device)
    init_cell = th.zeros(num_layers, num_envs, hidden_layer_size).to(device)
    for epoch in range(100000):
        prev_hidden, prev_cell = init_hidden.clone(), init_cell.clone()
        loss = 0
        if (epoch + 1) % 500 == 0:
            episode_length = max(episode_length - 1, 5)
        loss_list = th.zeros(episode_length * num_envs).to(device)
        prev_solution = env_maxcut.init(True)
        gamma0 = 0.98
        gamma = gamma0 ** episode_length
        for step in range(episode_length):
            #print(x_prev.shape)
            #print(x_prev.reshape(num_env, N, 1).shape)
            #x, h, c = opt_net(x_prev.reshape(num_env, N, 1), prev_h, prev_c)
            solution, hidden, cell = opt_net(prev_solution.reshape(num_envs, 1, num_nodes), prev_hidden, prev_cell)

            #x = x.reshape(num_env, N)
            l = env_maxcut.obj(solution.reshape(num_envs, num_nodes))
            loss_list[num_envs * (step):num_envs * (step + 1)] = l.detach()
            loss -= l.sum()
            #print(x_prev.shape, x.shape)
            l = env_maxcut.calc_obj_for_two_graphs_vmap(prev_solution.reshape(num_envs, num_nodes), solution.reshape(num_envs, num_nodes))
            loss -= 0.2 * l.sum()#max(0.05, (500-epoch) / 500) * l.sum()
            prev_solution = solution.detach()
            #prev_h, prev_c = h.detach(), c.detach()
            gamma /= gamma0

            if (step + 1) % 4 == 0:
                optimizer.zero_grad()
                #print(loss)
                loss.backward(retain_graph=True)
                optimizer.step()
                loss = 0
                #h, c = h_init.clone(), c_init.clone()
            prev_hidden, prev_cell = hidden.detach(), cell.detach()

        if epoch % 50 == 0:
            print(f"epoch:{epoch} | train:",  loss_list.max().item())
            hidden, cell = init_hidden, init_cell
            # print(h_init.mean(), c_init.mean())
            loss = 0
            #loss_list = []
            loss_list = th.zeros(episode_length * num_envs * 2).to(device)
            solution = env_maxcut.init(True)
            solutions = th.zeros(episode_length * num_envs * 2, num_nodes).to(device)
            for step in range(episode_length * 2):
                solution, hidden, cell = opt_net(solution.detach().reshape(num_envs, 1, num_nodes), hidden, cell)
                solution = solution.reshape(num_envs, num_nodes)
                solution2 = solution.detach()
                solution2 = (solution2>0.5).to(th.float32)
                # print(a)
                # assert 0
                l = env_maxcut.obj(solution2)
                loss_list[num_envs * (step):num_envs * (step + 1)] = l.detach()
                solutions[num_envs * step: num_envs * (step + 1)] = solution2.detach()
                #if (step + 6) % 2 == 0:
                    #optimizer.zero_grad()
                    #loss.backward()
                    #optimizer.step()
                    #loss = 0
                    #h, c = h_init.clone(), c_init.clone()
            val, idx = loss_list.max(dim=-1)
            result_file_name = calc_result_file_name(filename)
            local_write_result(solutions[idx], result_file_name)
            env_maxcut.best_solution = solutions[idx]
            print(f"epoch:{epoch} | test :",  loss_list.max().item())



if __name__ == "__main__":
    import sys

    filename = '../../data/gset/gset_14.txt'
    gpu_id = 0
    graph = read_nxgraph(filename)
    num_nodes = graph.number_of_nodes()
    hidden_layer_size = 4000
    learning_rate = 2e-5
    num_samples = 20
    episode_length = 30

    device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    th.manual_seed(7)
    opt_net = OptNet(num_nodes, hidden_layer_size).to(device)
    optimizer = th.optim.Adam(opt_net.parameters(), lr=learning_rate)

    train(filename, num_nodes, num_samples, device, opt_net, optimizer, episode_length, hidden_layer_size)