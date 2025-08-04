# dataset.py
"""TSP dataset generation and loading."""

import torch
from torch.utils.data import Dataset, DataLoader


class TSPDataset(Dataset):
    """Dataset for TSP instances."""
    
    def __init__(self, num_nodes, num_samples, random_seed=111):
        """
        Args:
            num_nodes: Number of nodes in each TSP instance
            num_samples: Number of TSP instances to generate
            random_seed: Random seed for reproducibility
        """
        super().__init__()
        
        # Set random seed for reproducibility
        generator = torch.Generator()
        generator.manual_seed(random_seed)
        
        # Generate random TSP instances
        self.data = []
        for _ in range(num_samples):
            # Generate random 2D coordinates in [0, 1] x [0, 1]
            nodes = torch.rand(num_nodes, 2, generator=generator)
            self.data.append(nodes)
            
        self.size = len(self.data)
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return idx, self.data[idx]


def create_data_loaders(args):
    """Create train and test data loaders.
    
    Args:
        args: Arguments containing dataset parameters
        
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        eval_loader: DataLoader for evaluation (full test set)
    """
    # Create datasets
    train_dataset = TSPDataset(args.seq_len, args.num_tr_dataset)
    test_dataset = TSPDataset(args.seq_len, args.num_te_dataset)
    
    # Determine if we should use pinned memory
    use_pin_memory = args.use_cuda and torch.cuda.is_available()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=use_pin_memory,
        num_workers=args.num_workers if hasattr(args, 'num_workers') else 0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=use_pin_memory,
        num_workers=args.num_workers if hasattr(args, 'num_workers') else 0
    )
    
    # Evaluation loader uses full test set
    eval_loader = DataLoader(
        test_dataset,
        batch_size=args.num_te_dataset,
        shuffle=False,
        pin_memory=use_pin_memory
    )
    
    return train_loader, test_loader, eval_loader, test_dataset