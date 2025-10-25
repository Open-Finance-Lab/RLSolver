import torch as th

INIT_TEMPERATURE = 1.0
FINAL_TEMPERATURE = 0
CHAIN_LENGTH = 200
BATCH_SIZE = 1
DATAPATH = "../../../rlsolver/data/syn_BA/BA_100_ID0.txt"
GPU_ID = 0

from rlsolver.methods.util import calc_device


DEVICE = calc_device(GPU_ID)
