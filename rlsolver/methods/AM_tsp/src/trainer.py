# trainer.py
"""Training logic for TSP solver."""

import torch
import torch.optim as optim
from tqdm import tqdm

from utils import moving_average, clip_grad_norm, AverageMeter, get_heuristic_solution


class TSPTrainer:
    """Trainer for TSP solver using REINFORCE algorithm."""
    
    def __init__(self, model, args):
        """
        Args:
            model: TSPSolver model
            args: Training arguments
        """
        self.model = model
        self.args = args
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        # Device
        self.device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Moving average baseline
        self.moving_avg = None
        self.beta = args.beta
        
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            avg_loss: Average loss for the epoch
            avg_reward: Average reward for the epoch
        """
        self.model.train()
        
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        
        for batch_idx, (indices, batch) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            batch = batch.to(self.device)
            batch_size = batch.size(0)
            
            # Forward pass
            rewards, log_probs, actions = self.model(batch)
            
            # Update moving average baseline
            if self.moving_avg is None:
                self.moving_avg = torch.zeros(len(train_loader.dataset), device=self.device)
                
            # Update baseline for current batch
            self.moving_avg[indices] = moving_average(
                self.moving_avg[indices], 
                rewards.detach(), 
                self.beta
            )
            
            # Calculate advantage
            advantage = rewards - self.moving_avg[indices]
            
            # REINFORCE loss
            log_probs_sum = log_probs.sum(dim=1)
            log_probs_sum = log_probs_sum.clamp(min=-100)  # Numerical stability
            loss = (advantage * log_probs_sum).mean()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()
            
            # Update meters
            loss_meter.update(loss.item(), batch_size)
            reward_meter.update(rewards.mean().item(), batch_size)
            
        return loss_meter.avg, reward_meter.avg
    
    def evaluate(self, eval_loader, heuristic_distances=None):
        """Evaluate model performance.
        
        Args:
            eval_loader: DataLoader for evaluation
            heuristic_distances: Pre-computed heuristic solutions
            
        Returns:
            avg_reward: Average tour length
            gap: Gap compared to heuristic (if available)
        """
        self.model.eval()
        
        all_rewards = []
        
        with torch.no_grad():
            for indices, batch in eval_loader:
                batch = batch.to(self.device)
                rewards, _, _ = self.model(batch)
                all_rewards.append(rewards.cpu())
                
        all_rewards = torch.cat(all_rewards)
        avg_reward = all_rewards.mean().item()
        
        # Calculate gap if heuristic solutions available
        gap = None
        if heuristic_distances is not None:
            ratio = all_rewards / heuristic_distances
            gap = ratio.mean().item()
            
        return avg_reward, gap
    
    def initialize_baseline(self, train_loader):
        """Initialize moving average baseline.
        
        Args:
            train_loader: DataLoader for training data
        """
        print("Initializing baseline...")
        self.model.eval()
        
        if self.moving_avg is None:
            self.moving_avg = torch.zeros(len(train_loader.dataset), device=self.device)
            
        with torch.no_grad():
            for indices, batch in tqdm(train_loader):
                batch = batch.to(self.device)
                rewards, _, _ = self.model(batch)
                self.moving_avg[indices] = rewards
                
    def train(self, train_loader, eval_loader, test_dataset=None):
        """Full training loop.
        
        Args:
            train_loader: DataLoader for training
            eval_loader: DataLoader for evaluation
            test_dataset: Test dataset for heuristic comparison
        """
        # Compute heuristic solutions if available
        heuristic_distances = None
        if test_dataset is not None:
            print("Computing heuristic solutions...")
            heuristic_distances = []
            for i, (_, pointset) in enumerate(tqdm(test_dataset)):
                dist = get_heuristic_solution(pointset)
                if dist is not None:
                    heuristic_distances.append(dist)
                else:
                    # If elkai not available, skip heuristic comparison
                    heuristic_distances = None
                    break
                    
            if heuristic_distances is not None:
                heuristic_distances = torch.tensor(heuristic_distances)
        
        # Initialize baseline
        self.initialize_baseline(train_loader)
        
        # Training loop
        for epoch in range(self.args.num_epochs):
            # Train
            avg_loss, avg_reward = self.train_epoch(train_loader, epoch)
            
            # Evaluate
            eval_reward, gap = self.evaluate(eval_loader, heuristic_distances)
            
            # Print results
            print(f"\n[Epoch {epoch}]")
            print(f"  Train Loss: {avg_loss:.4f}")
            print(f"  Train Reward: {avg_reward:.4f}")
            print(f"  Eval Reward: {eval_reward:.4f}")
            if gap is not None:
                print(f"  Gap vs Heuristic: {gap:.4f}x")
                
            # Save checkpoint if requested
            if hasattr(self.args, 'save_freq') and (epoch + 1) % self.args.save_freq == 0:
                from utils import save_checkpoint
                save_checkpoint(
                    self.model, 
                    self.optimizer, 
                    epoch, 
                    f"checkpoint_epoch_{epoch}.pt"
                )