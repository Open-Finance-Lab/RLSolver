import torch as th

INIT_TEMPERATURE = 1.0
FINAL_TEMPERATURE = 0.1
CHAIN_LENGTH = 10000
BATCH_SIZE = 1
GPU_ID = 0
K = 20
DATAPATH = '../../../rlsolver/data/tsplib/berlin52.tsp'
GPU_ID = 0

from rlsolver.methods.util import calc_device



DEVICE = calc_device(GPU_ID)
