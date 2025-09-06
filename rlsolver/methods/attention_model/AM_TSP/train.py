# train.py

import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from models import TSPActor
from dataset import create_distributed_data_loaders
from trainer import DistributedPOMOTrainer
from config import *
import config as args

# Global performance settings
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass
torch.backends.cudnn.benchmark = True


def setup_ddp(rank, world_size):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = MASTER_PORT
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """Clean up distributed training."""
    dist.destroy_process_group()


def main_worker(rank, world_size):
    """Main worker function for each GPU."""
    # Setup distributed training
    setup_ddp(rank, world_size)
    
    # Set random seeds
    torch.manual_seed(SEED + rank)
    torch.cuda.manual_seed(SEED + rank)
    
    # Print configuration from rank 0
    if rank == 0:
        print("="*50)
        print("TSP Solver with POMO (Policy Optimization with Multiple Optima)")
        print("="*50)
        print("\nConfiguration:")
        print(f"  GPU Configuration:")
        print(f"    - Mode: {'Multi-GPU' if MULTI_GPU_MODE else 'Single-GPU'}")
        print(f"    - World size: {world_size}")
        print(f"    - Training GPUs: {list(range(world_size))}")
        print(f"    - Inference GPU: {INFERENCE_GPU_ID}")
        print("\n  Model Configuration:")
        for key, value in vars(args).items():
            if not key.startswith('__') and not callable(value) and 'GPU' not in key:
                print(f"    {key}: {value}")
        print(f"  Algorithm: POMO")
        print(f"  POMO size: {NUM_TRAIN_ENVS}")
        print()
    
    # Create data loaders
    train_loader, test_loader, eval_loader, test_dataset = create_distributed_data_loaders(args, rank, world_size)
    
    # Create model
    model = TSPActor(
        embedding_size=EMBEDDING_SIZE,
        hidden_size=HIDDEN_SIZE,
        seq_len=NUM_NODES,
        n_head=N_HEAD,
        C=C
    ).cuda(rank)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    # Print model information
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        print(f"\nPOMO will generate {NUM_TRAIN_ENVS} rollouts per TSP instance")
        print(f"Each rollout starts from a different node (0 to {NUM_TRAIN_ENVS-1})")
        print(f"Shared baseline computed as mean of all {NUM_TRAIN_ENVS} rollouts")
        print()
    
    # Create POMO trainer
    trainer = DistributedPOMOTrainer(model, args, rank, world_size)
    
    # Train model with POMO
    trainer.train(train_loader, eval_loader, test_dataset)
    
    if rank == 0:
        print("\nTraining with POMO completed!")
    
    cleanup()


def train_single_gpu():
    """Training on single GPU (non-distributed)."""
    device = TRAIN_DEVICE
    
    print("="*50)
    print("TSP Solver with POMO (Single GPU Mode)")
    print("="*50)
    print(f"\nUsing device: {device}")
    
    # Set random seeds
    torch.manual_seed(SEED)
    if 'cuda' in device:
        torch.cuda.manual_seed(SEED)
        torch.cuda.set_device(TRAIN_GPU_ID)
    
    # Note: For single GPU, you would need to implement non-distributed versions
    # of the data loader and trainer. This is a simplified placeholder.
    raise NotImplementedError("Single GPU training not implemented. Use multi-GPU mode with world_size=1")


def main():
    """Main entry point."""
    if TRAIN_INFERENCE == 1:
        print("Inference mode selected. Please run inference.py instead.")
        return
    
    # Determine world size based on configuration
    if MULTI_GPU_MODE:
        world_size = get_num_gpus(True)
        if world_size == 0:
            raise RuntimeError("No GPUs available for training. Set USE_CUDA=False for CPU training.")
        
        print(f"Starting POMO distributed training on {world_size} GPUs")
        mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)
    else:
        # Single GPU training
        train_single_gpu()


if __name__ == '__main__':
    main()
