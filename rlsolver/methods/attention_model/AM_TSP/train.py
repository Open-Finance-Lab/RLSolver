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
    gpu_id = args.TRAIN_GPU_IDS[rank]
    torch.cuda.set_device(gpu_id)
    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup(world_size):
    """Clean up distributed training."""
    if world_size > 1:
        dist.destroy_process_group()


def main_worker(rank, world_size):
    """Main worker function for each GPU."""
    setup_ddp(rank, world_size)
    
    gpu_id = args.TRAIN_GPU_IDS[rank]
    
    torch.manual_seed(args.SEED + rank)
    torch.cuda.manual_seed(args.SEED + rank)

    if rank == 0:
        print("="*50)
        print("TSP Solver with POMO (Policy Optimization with Multiple Optima)")
        print("="*50)
        print("\nConfiguration:")
        for key, value in vars(args).items():
            if not key.startswith('__') and not callable(value):
                print(f"  {key}: {value}")
        print(f"  world_size: {world_size}")
        print()

    train_loader, test_loader, eval_loader, test_dataset = create_distributed_data_loaders(args, rank, world_size)

    model = TSPActor(
        embedding_size=args.EMBEDDING_SIZE,
        hidden_size=args.HIDDEN_SIZE,
        seq_len=args.NUM_NODES,
        n_head=args.N_HEAD,
        C=args.C
    ).cuda(gpu_id)

    if world_size > 1:
        model = DDP(model, device_ids=[gpu_id], find_unused_parameters=False)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        print()

    trainer = DistributedPOMOTrainer(model, args, rank, world_size)
    trainer.train(train_loader, eval_loader, test_dataset)

    if rank == 0:
        print("\nTraining with POMO completed!")

    cleanup(world_size)


def main():
    """Main entry point."""
    world_size = len(args.TRAIN_GPU_IDS)
    if world_size == 1:
        print(f"Starting POMO training on GPU {args.TRAIN_GPU_IDS[0]}")
        main_worker(0, 1)
    else:
        print(f"Starting POMO distributed training on GPUs {args.TRAIN_GPU_IDS}")
        mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()
