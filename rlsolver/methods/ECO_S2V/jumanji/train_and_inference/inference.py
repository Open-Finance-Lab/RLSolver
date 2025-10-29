import os
import sys

cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../../../')
sys.path.append(os.path.dirname(rlsolver_path))

import time
import torch
import networkx as nx

from rlsolver.methods.ECO_S2V.src.envs.inference_network_env import SpinSystemFactory
from rlsolver.methods.ECO_S2V.src.envs.util_envs_PECO import SetGraphGenerator
from rlsolver.methods.ECO_S2V.src.envs.util_envs import RewardSignal
from rlsolver.methods.ECO_S2V.src.envs.util_envs import ExtraAction
from rlsolver.methods.ECO_S2V.src.envs.util_envs import OptimisationTarget
from rlsolver.methods.ECO_S2V.src.envs.util_envs import SpinBasis
from rlsolver.methods.ECO_S2V.src.envs.util_envs import DEFAULT_OBSERVABLES
from rlsolver.methods.util_read_data import read_nxgraph

from rlsolver.methods.ECO_S2V.jumanji.agents.AgentPPO import AgentA2C
from rlsolver.methods.ECO_S2V.config import *
from rlsolver.methods.util_write_read_result import write_graph_result
from rlsolver.methods.ECO_S2V.jumanji.train.config import Config


def run(graph_folder, num_envs, mini_envs):
    print("\n----- Running {} -----\n".format(os.path.basename(__file__)))
    network_save_path = NEURAL_NETWORK_SAVE_PATH

    print("Testing network: ", network_save_path)
    env_args_ = {
        'env_name': 'maxcut',
        'num_envs': NUM_TRAIN_ENVS,
        'num_nodes': NUM_TRAIN_NODES,
        'state_dim': 2,
        'action_dim': 1,
        'if_discrete': True,
    }
    args = Config(AgentA2C, "maxcut", env_args_)
    args.gpu_id = INFERENCE_GPU_ID
    agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)

    # agent = AgentA2C(device=TRAIN_DEVICE, num_envs=NUM_TRAIN_ENVS,)
    agent.act.load_state_dict(torch.load(network_save_path, map_location=INFERENCE_DEVICE))
    for param in agent.act.parameters():
        param.requires_grad = False
    agent.act.eval()
    env_args = {'observables': DEFAULT_OBSERVABLES,
                'reward_signal': RewardSignal.BLS,
                'extra_action': ExtraAction.NONE,
                'optimisation_target': OptimisationTarget.CUT,
                'spin_basis': SpinBasis.BINARY,
                'norm_rewards': True,
                'memory_length': None,
                'horizon_length': None,
                'stag_punishment': None,
                'basin_reward': 1. / NUM_TRAIN_NODES,
                'reversible_spins': True,
                }
    step_factor = 2
    files = os.listdir(graph_folder)

    for prefix in INFERENCE_PREFIXES:
        graphs = []
        file_list = []
        for file in files:
            if prefix in file:
                file = os.path.join(graph_folder, file).replace("\\", "/")
                file_list.append(file)
                # g = load_graph_from_txt(file)
                g = read_nxgraph(file)
                g_array = nx.to_numpy_array(g)
                g_tensor = torch.tensor(g_array, dtype=torch.float, device=INFERENCE_DEVICE)
                graphs.append(g_tensor)
        start_time = time.time()
        if len(graphs) > 0:
            for i, graph_tensor in enumerate(graphs):
                test_graph_generator = SetGraphGenerator(graph_tensor, device=INFERENCE_DEVICE)

                best_obj = float('-inf')
                best_sol = None

                num_batches = (num_envs + mini_envs - 1) // mini_envs  # 计算分批次数
                for batch in range(num_batches):
                    current_mini_sims = min(mini_envs, num_envs - batch * mini_envs)  # 防止超出 num_envs

                    test_env = SpinSystemFactory.get(
                        test_graph_generator,
                        graph_tensor.shape[0] * step_factor,
                        **env_args,
                        device=INFERENCE_DEVICE,
                        num_envs=current_mini_sims,  # 只处理 mini_sims 个环境
                    )
                    agent.num_envs = current_mini_sims
                    agent.n_spins = graph_tensor.shape[0]
                sol = agent.inference(env=test_env, max_steps=graph_tensor.shape[0] * step_factor)
                obj, obj_index = torch.max(test_env.best_score, dim=0)
                if obj > best_obj:
                    best_obj = obj.item()
                    best_sol = test_env.best_spins[obj_index]
                run_duration = time.time() - start_time
                best_sol = ((best_sol + 1) / 2).to(torch.int).cpu().numpy()
                write_graph_result(best_obj, run_duration, best_sol.shape[0], 'jumanji', best_sol, file_list[i], plus1=False)


def inference(agent, env, stpes):
    env.reset()
    agent._explore_vec_env(env=env, horizon_len=stpes)
    return env.best_score
