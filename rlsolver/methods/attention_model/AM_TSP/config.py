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
# Training
MULTI_GPU_MODE = True  # Whether to use all available GPUs
TRAIN_GPU_ID = 0  # GPU ID when not using multi-GPU mode
TRAIN_DEVICE = torch.device(f'cuda:{TRAIN_GPU_ID}' if torch.cuda.is_available() else 'cpu')

# Inference
INFERENCE_GPU_ID = 0  # GPU for inference/evaluation
INFERENCE_DEVICE = torch.device(f'cuda:{INFERENCE_GPU_ID}' if torch.cuda.is_available() else 'cpu')

# ================== Distributed Training ==================
MASTER_ADDR = 'localhost'
MASTER_PORT = '12355'

# ================== Helper Functions ==================
def get_num_gpus():
    """Get number of available GPUs."""
    return torch.cuda.device_count()

def get_world_size():
    """Get world size for distributed training."""
    if MULTI_GPU_MODE:
        return get_num_gpus()
    return 1
