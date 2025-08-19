# config.py

# ================== Model Configuration ==================
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 128
N_HEAD = 8
C = 15.0
SEQ_LEN = 20  # TSP problem size

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
NUM_TRAIN_ENVS = SEQ_LEN  # Number of parallel rollouts during training (POMO uses problem size)

# ================== Inference Configuration ==================
INFERENCE_BATCH_SIZE = 64  # Batch size for inference
NUM_INFERENCE_ENVS = SEQ_LEN  # Number of parallel rollouts during inference
COMPUTE_HEURISTIC_GAP = True  # Whether to compute gap vs heuristic solver
SAVE_RESULTS = True  # Whether to save inference results

# ================== Paths ==================
# Model paths
MODEL_PATH = "model.pth"  # Path to trained model for inference
CHECKPOINT_DIR = "checkpoints"

# Results paths
RESULTS_DIR = "results"
RESULTS_FILENAME = "inference_results.json"

# ================== System Configuration ==================
USE_CUDA = True
NUM_WORKERS = 0  # DataLoader workers (increase based on CPU cores)
SEED = 111

# ================== Distributed Training ==================
MASTER_ADDR = 'localhost'
MASTER_PORT = '12355'