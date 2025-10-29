import os
import sys
import time

cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../../')
sys.path.append(os.path.dirname(rlsolver_path))
from rlsolver.methods.ECO_S2V.config import *

start_time = time.time()
if TRAIN_INFERENCE == 0:
    if ALG == Alg.eco:
        from rlsolver.methods.ECO_S2V.train_and_inference.train_ECO import run
    elif ALG == Alg.s2v:
        from rlsolver.methods.ECO_S2V.train_and_inference.train_S2V import run
    elif ALG == Alg.eco_torch:
        from rlsolver.methods.ECO_S2V.train_and_inference.train_ECO_torch import run
    elif ALG == Alg.peco:
        from rlsolver.methods.ECO_S2V.train_and_inference.train_PECO import run
    elif ALG == Alg.jumanji:
        from rlsolver.methods.ECO_S2V.jumanji.train_and_inference.train import run
    elif ALG == Alg.rl4co:
        from rlsolver.methods.ECO_S2V.rl4co.train import run
    else:
        raise ValueError('Algorithm not recognized')
    run(save_loc=NEURAL_NETWORK_DIR)

if TRAIN_INFERENCE == 1:
    if ALG == Alg.peco:
        from rlsolver.methods.ECO_S2V.train_and_inference.inference_PECO import run

        run(graph_folder=DATA_DIR,
            num_envs=NUM_INFERENCE_ENVS,
            mini_sims=MINI_INFERENCE_ENVS)
    elif ALG == Alg.eco or ALG == Alg.s2v:
        from rlsolver.methods.ECO_S2V.train_and_inference.inference import run

        run(save_loc=NEURAL_NETWORK_DIR, graph_save_loc=DATA_DIR, network_save_path=NEURAL_NETWORK_SAVE_PATH,
            batched=True, max_batch_size=None, max_parallel_jobs=1, prefixes=INFERENCE_PREFIXES)
    elif ALG == Alg.jumanji:
        from rlsolver.methods.ECO_S2V.jumanji.train_and_inference.inference import run

        run(graph_folder=DATA_DIR, num_envs=NUM_INFERENCE_ENVS, mini_envs=MINI_INFERENCE_ENVS)
    elif ALG == Alg.rl4co:
        from rlsolver.methods.ECO_S2V.rl4co.inference import run

        run(graph_dir=RL4CO_GRAPH_DIR, num_envs=NUM_INFERENCE_ENVS)
    else:
        raise ValueError('Algorithm not recognized')
running_duration = time.time() - start_time
print("running_duration: ", running_duration)
