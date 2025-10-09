"""Configuration for TSP Solver."""

import torch
from rlsolver.methods.util import calc_device

# ================== Model Configuration ==================
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 128
N_HEAD = 8
C = 15.0

TRAIN_INFERENCE = 0  # 0: train, 1: inference
NUM_NODES = 20

# ================== Dataset Configuration ==================
NUM_TR_DATASET = 100000
NUM_TE_DATASET = 200
NUM_TEST_SAMPLES = 1000
TEST_SEED = 222

# ================== Training Configuration ==================
NUM_EPOCHS = 10
BATCH_SIZE = 256
LR = 0.0002
GRAD_CLIP = 1.5

NUM_TRAIN_ENVS = NUM_NODES  # POMO size

# ================== Inference Configuration ==================
INFERENCE_BATCH_SIZE = 64
NUM_INFERENCE_ENVS = NUM_NODES
COMPUTE_HEURISTIC_GAP = True
SAVE_RESULTS = True

# ================== Device Configuration ==================
MULTI_GPU_MODE = True
TRAIN_GPU_ID = 0
TRAIN_DEVICE = calc_device(TRAIN_GPU_ID)

INFERENCE_GPU_ID = 0
INFERENCE_DEVICE = calc_device(INFERENCE_GPU_ID)

# ================== Paths ==================
MODEL_PATH = "model.pth"
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"
RESULTS_FILENAME = "inference_results.json"

# ================== System Configuration ==================
NUM_WORKERS = 0
SEED = 111

# ================== Distributed Training ==================
MASTER_ADDR = 'localhost'
MASTER_PORT = '12355'


def get_num_gpus(use_cuda: bool):
    """Get number of available GPUs."""
    if not use_cuda:
        return 0
    return torch.cuda.device_count()