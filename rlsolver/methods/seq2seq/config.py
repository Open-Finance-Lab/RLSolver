import torch

NUM_NODES = 10000
DATA_PATH = 'data/test_data'
GPU_ID = 0
BATCH_SIZE = 2**5

from rlsolver.methods.util import calc_device
DEVICE = calc_device(GPU_ID)

