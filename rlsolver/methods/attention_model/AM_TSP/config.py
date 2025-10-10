import torch
from rlsolver.methods.util import calc_device
# ================== Model Configuration ==================
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 128
N_HEAD = 8
C = 15.0

TRAIN_INFERENCE = 0  # 0: train, 1: inference
assert TRAIN_INFERENCE in [0, 1]
NUM_NODES = 20  # TSP problem size

# ================== Dataset Configuration ==================
NUM_TR_DATASET = 100000
NUM_TE_DATASET = 200
NUM_TEST_SAMPLES = 1000  # For inference testing
TEST_SEED = 222  # Different seed for test set

# ================== Training Configuration ==================
NUM_EPOCHS = 10
BATCH_SIZE = 256
LR = 0.0002
GRAD_CLIP = 1.5
BETA = 0.9  # Legacy, kept for compatibility

# POMO specific - Training
NUM_TRAIN_ENVS = NUM_NODES  # Number of parallel rollouts during training (POMO uses problem size)

# ================== Inference Configuration ==================
INFERENCE_BATCH_SIZE = 64  # Batch size for inference
NUM_INFERENCE_ENVS = NUM_NODES  # Number of parallel rollouts during inference
COMPUTE_HEURISTIC_GAP = True  # Whether to compute gap vs heuristic solver
SAVE_RESULTS = True  # Whether to save inference results



# Training
MULTI_GPU_MODE = False  # Whether to use all available GPUs
TRAIN_GPU_ID = 0  # GPU ID when not using multi-GPU mode
TRAIN_DEVICE = calc_device(TRAIN_GPU_ID)

# Inference
INFERENCE_GPU_ID = 0  # GPU for inference/evaluation
INFERENCE_DEVICE = calc_device(INFERENCE_GPU_ID)

# ================== Paths ==================
# Model paths
MODEL_PATH = "model.pth"  # Path to trained model for inference
CHECKPOINT_DIR = "checkpoints"

# Results paths
RESULTS_DIR = "results"
RESULTS_FILENAME = "inference_results.json"

# ================== System Configuration ==================
NUM_WORKERS = 0  # DataLoader workers (increase based on CPU cores)
SEED = 111

# ================== Distributed Training ==================
MASTER_ADDR = 'localhost'
MASTER_PORT = '12355'

# ================== Device Mapping ==================


def get_num_gpus(use_cuda: bool):
    """Get number of available GPUs."""
    if not use_cuda:
        return 0
    return torch.cuda.device_count()
