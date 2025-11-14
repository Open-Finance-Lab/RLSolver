# dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


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
        generator = torch.Generator()
        generator.manual_seed(random_seed)
        self.data = []
        for _ in range(num_samples):
            nodes = torch.rand(num_nodes, 2, generator=generator)
            self.data.append(nodes)
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return idx, self.data[idx]


def create_distributed_data_loaders(args, rank, world_size):
    """Create distributed train and test data loaders.
    
    Args:
        args: Arguments containing dataset parameters
        rank: Current process rank
        world_size: Total number of processes
        
    Returns:
        train_loader: Distributed DataLoader for training data
        test_loader: DataLoader for test data
        eval_loader: DataLoader for evaluation (full test set)
        test_dataset: Test dataset for heuristic comparison
    """
    train_dataset = TSPDataset(args.NUM_NODES, args.NUM_TR_DATASET, args.SEED)
    test_dataset = TSPDataset(args.NUM_NODES, args.NUM_TE_DATASET, args.SEED)

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    per_gpu_num_envs = args.NUM_ENVS // world_size
    if per_gpu_num_envs < 1:
        per_gpu_num_envs = 1
        if rank == 0:
            print(f"Warning: NUM_ENVS {args.NUM_ENVS} too small for {world_size} GPUs, using 1 per GPU")

    common_kwargs = {
        "pin_memory": True,
        "num_workers": args.NUM_WORKERS,
        "persistent_workers": args.NUM_WORKERS > 0,
    }

    if args.NUM_WORKERS > 0:
        common_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_gpu_num_envs,
        sampler=train_sampler,
        drop_last=True,
        **common_kwargs
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.INFERENCE_BATCH_SIZE,
        shuffle=False,
        **common_kwargs
    )

    eval_loader = DataLoader(
        test_dataset,
        batch_size=args.NUM_TE_DATASET,
        shuffle=False,
        **common_kwargs
    )

    return train_loader, test_loader, eval_loader, test_dataset
