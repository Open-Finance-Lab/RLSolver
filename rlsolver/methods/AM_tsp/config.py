"""Configuration for TSP solver."""

# Model parameters
embedding_size = 128
hidden_size = 128
n_head = 4
C = 10.0

# Dataset parameters
seq_len = 30
num_tr_dataset = 10000
num_te_dataset = 2000

# Training parameters
num_epochs = 500
batch_size = 64
lr = 0.0003
grad_clip = 1.5
beta = 0.9

# System parameters
use_cuda = True
num_workers = 0
seed = 111

# Saving parameters
save_freq = 10