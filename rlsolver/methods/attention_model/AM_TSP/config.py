# config.py

import torch

# ================== Model Configuration ==================
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 128
N_HEAD = 8
C = 15.0
NUM_NODES = 20  # TSP problem size

# ================== Dataset Configuration ==================
NUM_TR_DATASET = 100000
NUM_TE_DATASET = 10

# ================== Training Configuration ==================
NUM_EPOCHS = 1000
NUM_ENVS = 1024  # Number of TSP instances to sample per training step
BATCH_SIZE = 4096  # Batch size for gradient computation 
LR = 0.0002
GRAD_CLIP = 1.5

# POMO specific - Training
NUM_POMO = NUM_NODES  # Number of POMO rollouts per instance (equals NUM_NODES)

# ================== Inference Configuration ==================
INFERENCE_BATCH_SIZE = 4096
NUM_INFERENCE_POMO = NUM_NODES  # Number of POMO rollouts per instance during inference
NUM_TEST_SAMPLES = 1000
TEST_SEED = 1234
COMPUTE_HEURISTIC_GAP = True
SAVE_RESULTS = True

# ================== Paths ==================
MODEL_PATH = "model.pth"
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"

# ================== System Configuration ==================
NUM_WORKERS = 0
SEED = 111

# ================== GPU Configuration ==================
# Training GPU IDs - list of GPU IDs to use for training
# Single GPU: TRAIN_GPU_IDS = [0]
# Multi-GPU:  TRAIN_GPU_IDS = [0, 1, 2, 3]
TRAIN_GPU_IDS = [0]

# Inference
INFERENCE_GPU_ID = 0
INFERENCE_DEVICE = torch.device(f'cuda:{INFERENCE_GPU_ID}' if torch.cuda.is_available() else 'cpu')

# ================== Distributed Training ==================
MASTER_ADDR = 'localhost'
MASTER_PORT = '12355'
