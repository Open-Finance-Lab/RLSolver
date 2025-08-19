# train.py

import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from models import TSPActor
from dataset import create_distributed_data_loaders
from trainer import DistributedPOMOTrainer
import config as args

# Global performance settings
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass
torch.backends.cudnn.benchmark = True


def setup_ddp(rank, world_size):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = args.MASTER_ADDR
    os.environ['MASTER_PORT'] = args.MASTER_PORT
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
    torch.manual_seed(args.SEED + rank)
    torch.cuda.manual_seed(args.SEED + rank)
    
    # Print configuration from rank 0
    if rank == 0:
        print("="*50)
        print("TSP Solver with POMO (Policy Optimization with Multiple Optima)")
        print("="*50)
        print("\nConfiguration:")
        for key, value in vars(args).items():
            print(f"  {key}: {value}")
        print(f"  world_size: {world_size}")
        print(f"  algorithm: POMO")
        print(f"  pomo_size: {args.NUM_TRAIN_ENVS}")
        print()
    
    # Create data loaders
    train_loader, test_loader, eval_loader, test_dataset = create_distributed_data_loaders(args, rank, world_size)
    
    # Create model
    model = TSPActor(
        embedding_size=args.EMBEDDING_SIZE,
        hidden_size=args.HIDDEN_SIZE,
        seq_len=args.SEQ_LEN,
        n_head=args.N_HEAD,
        C=args.C
    ).cuda(rank)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    # Print model information
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        print(f"\nPOMO will generate {args.NUM_TRAIN_ENVS} rollouts per TSP instance")
        print(f"Each rollout starts from a different node (0 to {args.NUM_TRAIN_ENVS-1})")
        print(f"Shared baseline computed as mean of all {args.NUM_TRAIN_ENVS} rollouts")
        print()
    
    # Create POMO trainer
    trainer = DistributedPOMOTrainer(model, args, rank, world_size)
    
    # Train model with POMO
    trainer.train(train_loader, eval_loader, test_dataset)
    
    if rank == 0:
        print("\nTraining with POMO completed!")
    
    cleanup()


def main():
    """Main entry point."""
    world_size = torch.cuda.device_count()
    
    if world_size == 0:
        raise RuntimeError("No GPUs available for training")
    
    print(f"Starting POMO distributed training on {world_size} GPUs")
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()