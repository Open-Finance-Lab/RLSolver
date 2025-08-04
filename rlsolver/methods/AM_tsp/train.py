"""Main training script for TSP solver."""

import torch

from RLSolver.rlsolver.methods.AM_tsp.src.models import TSPSolver
from dataset import create_data_loaders
from RLSolver.rlsolver.methods.AM_tsp.src.trainer import TSPTrainer
import config as args


def main():
    """Main training function."""
    
    # Set random seeds
    torch.manual_seed(args.seed)
    if args.use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Print configuration
    print("Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print()
    
    # Check CUDA availability
    if args.use_cuda and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.use_cuda = False
    
    # Create data loaders
    train_loader, test_loader, eval_loader, test_dataset = create_data_loaders(args)
    
    # Create model
    model = TSPSolver(
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        seq_len=args.seq_len,
        n_head=args.n_head,
        C=args.C
    )
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print()
    
    # Create trainer
    trainer = TSPTrainer(model, args)
    
    # Train model
    trainer.train(train_loader, eval_loader, test_dataset)
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()