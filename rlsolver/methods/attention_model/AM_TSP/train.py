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
    if world_size == 1:
        gpu_id = args.TRAIN_GPU_ID
    else:
        gpu_id = rank
    torch.cuda.set_device(gpu_id)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """Clean up distributed training."""
    dist.destroy_process_group()


def main_worker(rank, world_size):
    """Main worker function for each GPU."""
    setup_ddp(rank, world_size)
    
    if world_size == 1:
        gpu_id = args.TRAIN_GPU_ID
    else:
        gpu_id = rank
    
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

    cleanup()


def main():
    """Main entry point."""
    if args.MULTI_GPU_MODE:
        world_size = args.get_num_gpus()
        if world_size == 0:
            raise RuntimeError("No GPUs available for training")
        print(f"Starting POMO distributed training on {world_size} GPUs")
        mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)
    else:
        print(f"Starting POMO training on GPU {args.TRAIN_GPU_ID}")
        main_worker(0, 1)
        cleanup()


if __name__ == '__main__':
    main()
