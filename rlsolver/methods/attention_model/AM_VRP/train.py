import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from time import gmtime, strftime

from config import Config
from model import AttentionDynamicModel
from reinforce_baseline import RolloutBaseline, validate
from utils import generate_data_onfly_distributed, create_data_on_disk, get_cur_time


def setup_distributed(rank, world_size):
    """Initialize distributed training environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def train_model(config, optimizer, model, baseline, validation_dataset):
    """Train the VRP model using REINFORCE with baseline."""
    
    # Get unwrapped model reference
    model_unwrapped = model.module if hasattr(model, 'module') else model
    model_unwrapped.train()
    
    for epoch in range(config.EPOCHS):
        if config.RANK == 0:
            print(f"\n{get_cur_time()} Epoch {epoch}/{config.EPOCHS}")
        
        # Generate distributed training data
        dataset = generate_data_onfly_distributed(
            num_samples=config.SAMPLES,
            graph_size=config.GRAPH_SIZE,
            batch_size=config.BATCH_SIZE,
            rank=config.RANK,
            world_size=config.WORLD_SIZE,
            device=config.DEVICE
        )
        
        # Pre-compute baseline values
        bl_vals = baseline.eval_all(dataset)
        if bl_vals is not None:
            bl_vals = bl_vals.view(-1, config.BATCH_SIZE)
        
        epoch_loss_sum = torch.tensor(0.0, device=config.DEVICE)
        epoch_cost_sum = torch.tensor(0.0, device=config.DEVICE)
        num_batches = 0
        
        # Training loop
        model.train()
        for num_batch, batch in enumerate(tqdm(dataset, desc=f"Training GPU {config.RANK}", disable=config.RANK != 0)):
            batch = [b.to(config.DEVICE) for b in batch]
            
            # Forward pass
            cost, log_likelihood = model(batch)
            
            # Get baseline value
            if bl_vals is not None:
                bl_val = bl_vals[num_batch]
            else:
                bl_val = baseline.eval(batch, cost)
            
            # REINFORCE loss
            advantage = cost - bl_val.detach()
            loss = (advantage * log_likelihood).mean()
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            # Gradient clipping before DDP averaging
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            
            optimizer.step()
            
            # Track metrics
            epoch_loss_sum += loss.detach()
            epoch_cost_sum += cost.mean().detach()
            num_batches += 1
        
        # Compute epoch averages
        epoch_loss_avg = epoch_loss_sum / num_batches
        epoch_cost_avg = epoch_cost_sum / num_batches
        
        # Reduce metrics across all GPUs
        if dist.is_initialized():
            dist.all_reduce(epoch_loss_avg, op=dist.ReduceOp.SUM)
            dist.all_reduce(epoch_cost_avg, op=dist.ReduceOp.SUM)
            epoch_loss_avg /= config.WORLD_SIZE
            epoch_cost_avg /= config.WORLD_SIZE
        
        # Update baseline
        baseline.epoch_callback(model_unwrapped, epoch)
        model_unwrapped.set_decode_type("sampling")
        
        # Synchronize before validation
        if dist.is_initialized():
            dist.barrier()
        
        # Validation
        val_cost = validate(validation_dataset, model, config.VAL_BATCH_SIZE, config.DEVICE, config.RANK)
        
        if config.RANK == 0:
            print(f"Summary - Loss: {epoch_loss_avg.item():.4f}, Cost: {epoch_cost_avg.item():.4f}, Val: {val_cost:.4f}")
    
    # Save final model (only from rank 0)
    if config.RANK == 0:
        torch.save(model_unwrapped.state_dict(), 'model.pth')
        print(f"Model saved to model.pth")


def train_worker(rank, world_size, config):
    """Worker function for distributed training."""
    setup_distributed(rank, world_size)
    
    config.RANK = rank
    config.WORLD_SIZE = world_size
    config.LOCAL_RANK = rank
    
    device = torch.device(f'cuda:{rank}')
    config.DEVICE = device
    
    if rank == 0:
        print(f'Using {world_size} GPU(s) for training')
        print(f'Batch size per GPU: {config.BATCH_SIZE}')
        print(f'Total effective batch size: {config.BATCH_SIZE * world_size}')
    
    # Set random seed
    torch.manual_seed(config.SEED + rank)
    torch.cuda.manual_seed_all(config.SEED + rank)
    
    # Create model
    model = AttentionDynamicModel(
        embedding_dim=config.EMBEDDING_DIM,
        n_encode_layers=config.N_ENCODE_LAYERS,
        n_heads=config.N_HEADS,
        tanh_clipping=config.TANH_CLIPPING
    ).to(device)
    
    model.set_decode_type("sampling")
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    # Create optimizer (scale learning rate by world size)
    lr = config.LR * world_size
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Create validation dataset
    if rank == 0:
        print('Creating validation dataset...')
    val_dataset = create_data_on_disk(
        graph_size=config.GRAPH_SIZE,
        num_samples=config.VAL_SIZE,
        filename=None,  # 不保存验证集文件
        seed=config.SEED,
        rank=rank
    )
    
    # Create baseline
    baseline = RolloutBaseline(
        model=model.module,
        config=config,
        epoch=0
    )
    
    # Train model
    if rank == 0:
        print('Starting training...')
    
    try:
        train_model(config, optimizer, model, baseline, val_dataset)
        if rank == 0:
            print('Training completed!')
    finally:
        cleanup_distributed()


def main():
    """Main function that automatically uses all available GPUs."""
    config = Config()
    
    # Automatically detect number of GPUs
    world_size = torch.cuda.device_count()
    
    if world_size == 0:
        raise RuntimeError("No GPUs available. This code requires at least one GPU.")
    
    print(f"Detected {world_size} GPU(s)")
    
    if world_size == 1:
        # Single GPU mode
        print("Running in single GPU mode")
        config.RANK = 0
        config.WORLD_SIZE = 1
        config.LOCAL_RANK = 0
        
        device = torch.device('cuda:0')
        config.DEVICE = device
        
        # Set random seed
        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed_all(config.SEED)
        
        # Create model
        model = AttentionDynamicModel(
            embedding_dim=config.EMBEDDING_DIM,
            n_encode_layers=config.N_ENCODE_LAYERS,
            n_heads=config.N_HEADS,
            tanh_clipping=config.TANH_CLIPPING
        ).to(device)
        
        model.set_decode_type("sampling")
        
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=config.LR)
        
        # Create validation dataset
        print('Creating validation dataset...')
        val_dataset = create_data_on_disk(
            graph_size=config.GRAPH_SIZE,
            num_samples=config.VAL_SIZE,
            filename=None,
            seed=config.SEED,
            rank=0
        )
        
        # Create baseline
        baseline = RolloutBaseline(
            model=model,
            config=config,
            epoch=0
        )
        
        # Train model
        print('Starting training...')
        train_model(config, optimizer, model, baseline, val_dataset)
        print('Training completed!')
        
    else:
        # Multi-GPU mode
        print(f"Running in multi-GPU mode with {world_size} GPUs")
        
        mp.spawn(
            train_worker,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )


if __name__ == '__main__':
    if torch.cuda.is_available():
        mp.set_start_method('spawn', force=True)
    
    main()