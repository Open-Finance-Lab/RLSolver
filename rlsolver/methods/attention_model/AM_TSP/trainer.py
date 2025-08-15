"""TSP Trainer with distributed training support and vmap optimizations."""

import os
import json
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from datetime import datetime

from env import TSPEnv, VectorizedTSPEnv
from utils import moving_average, clip_grad_norm, AverageMeter, get_heuristic_solution


def rollout_episode_vmap(model, nodes, device='cuda'):
    """Vectorized rollout for TSP using vmap optimizations.
    
    Args:
        model: TSPActor model (unwrapped)
        nodes: [batch_size, seq_len, 2]
    
    Returns:
        tour_length: [batch_size]
        log_probs: [batch_size, seq_len]
        actions: [batch_size, seq_len]
    """
    batch_size = nodes.size(0)
    seq_len = nodes.size(1)
    
    # Pre-compute encoder embeddings once
    with torch.no_grad():
        embedded = model.network.embedding(nodes)
        encoded = model.network.encoder(embedded)
    
    # Initialize state tensors
    visited_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    current_node = None
    first_node = None
    
    log_probs_list = []
    actions_list = []
    
    # Vectorized rollout loop
    for step in range(seq_len):
        # Build observation
        obs = {
            'nodes': nodes,
            'current_node': current_node,
            'first_node': first_node,
            'action_mask': ~visited_mask,
            'encoded': encoded
        }
        
        with torch.no_grad():
            # Get actions for all environments in parallel
            action, log_prob = model.get_action(obs, deterministic=False)
        
        # Update states using vectorized operations
        if current_node is None:
            first_node = action.clone()
        current_node = action
        
        # Vectorized mask update
        batch_indices = torch.arange(batch_size, device=device)
        visited_mask[batch_indices, action] = True
        
        log_probs_list.append(log_prob)
        actions_list.append(action)
    
    # Stack results
    log_probs = torch.stack(log_probs_list, dim=1)
    actions = torch.stack(actions_list, dim=1)
    
    # Compute tour lengths using vectorized environment
    vec_env = VectorizedTSPEnv(nodes, device=device)
    tour_lengths = vec_env.compute_all_tours(actions)
    
    return tour_lengths, log_probs, actions


def rollout_episode(model, nodes, device='cuda'):
    """Perform a complete rollout for TSP (original implementation for compatibility).
    
    Args:
        model: TSPActor model (unwrapped)
        nodes: [batch_size, seq_len, 2] or [seq_len, 2]
        
    Returns:
        tour_length: [batch_size]
        log_probs: [batch_size, seq_len]
        actions: [batch_size, seq_len]
    """
    # Use vectorized version
    return rollout_episode_vmap(model, nodes, device)


class DistributedTSPTrainer:
    """Distributed trainer for non-autoregressive TSP solver with vmap optimizations."""
    
    def __init__(self, model, args, rank, world_size):
        self.model = model
        # Extract the actual model from DDP wrapper
        self.model_module = model.module if hasattr(model, 'module') else model
        self.args = args
        self.rank = rank
        self.world_size = world_size
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=args.LR)
        
        # Mixed precision
        self.scaler = GradScaler()
        
        # Moving average baseline
        self.moving_avg = None
        self.beta = args.BETA
        self.baseline_sync_freq = 500
        self.step_count = 0
        
        # Use vmap for rollouts
        self.use_vmap = True
        
        # Best model tracking
        self.best_eval_reward = float('inf')  # Lower is better for TSP
        self.best_epoch = -1
        
        # Create checkpoint directory
        self.checkpoint_dir = "checkpoints"
        if self.rank == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def save_config(self):
        """Save training configuration to checkpoint directory."""
        if self.rank == 0:
            config = {
                'model_config': {
                    'embedding_size': self.model_module.network.embedding_size,
                    'hidden_size': self.model_module.network.hidden_size,
                    'seq_len': self.model_module.network.seq_len,
                    'n_head': self.model_module.network.n_head,
                    'C': self.model_module.network.C,
                },
                'training_config': {
                    'lr': self.args.LR,
                    'num_epochs': self.args.NUM_EPOCHS,
                    'beta': self.args.BETA,
                    'grad_clip': self.args.GRAD_CLIP,
                    'batch_size': getattr(self.args, 'BATCH_SIZE', None),
                },
                'timestamp': datetime.now().isoformat(),
                'use_vmap': self.use_vmap,
            }
            
            config_path = os.path.join(self.checkpoint_dir, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"Configuration saved to {config_path}")
    
    def sync_baseline(self):
        """Synchronize baselines across GPUs."""
        if self.moving_avg is not None:
            baseline_sum = self.moving_avg.clone()
            dist.all_reduce(baseline_sum, op=dist.ReduceOp.SUM)
            self.moving_avg = baseline_sum / self.world_size
    
    def compute_loss_vmap(self, nodes, indices):
        """Compute REINFORCE loss with baseline using vmap optimizations.
        
        Args:
            nodes: [batch_size, seq_len, 2]
            indices: [batch_size] dataset indices
            
        Returns:
            loss: scalar tensor
            rewards: [batch_size] tour lengths
        """
        batch_size = nodes.size(0)
        seq_len = nodes.size(1)
        device = nodes.device
        
        # Initialize states
        visited_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        current_node = None
        first_node = None
        
        log_probs_list = []
        actions_list = []
        
        # Rollout with gradients
        for step in range(seq_len):
            obs = {
                'nodes': nodes,
                'current_node': current_node,
                'first_node': first_node,
                'action_mask': ~visited_mask
            }
            
            # Get action from model (with gradients)
            action, log_prob = self.model_module.get_action(obs, deterministic=False)
            
            # Update states
            if current_node is None:
                first_node = action.clone()
            current_node = action
            
            # Vectorized mask update
            batch_indices = torch.arange(batch_size, device=device)
            visited_mask[batch_indices, action] = True
            
            log_probs_list.append(log_prob)
            actions_list.append(action)
        
        # Stack results
        log_probs = torch.stack(log_probs_list, dim=1)
        actions = torch.stack(actions_list, dim=1)
        
        # Compute rewards using vectorized environment
        vec_env = VectorizedTSPEnv(nodes, device=device)
        rewards = vec_env.compute_all_tours(actions)
        
        # Initialize moving average if needed
        if self.moving_avg is None:
            self.moving_avg = torch.zeros(len(indices), device=device)
        
        # Update baseline
        with torch.no_grad():
            self.moving_avg[indices] = moving_average(
                self.moving_avg[indices],
                rewards.detach(),
                self.beta
            )
        
        # Calculate advantage
        advantage = rewards - self.moving_avg[indices]
        
        # REINFORCE loss
        log_probs_sum = log_probs.sum(dim=1)
        log_probs_sum = log_probs_sum.clamp(min=-100)
        loss = (advantage * log_probs_sum).mean()
        
        return loss, rewards
    
    def compute_loss(self, nodes, indices):
        """Compute REINFORCE loss with baseline.
        
        Args:
            nodes: [batch_size, seq_len, 2]
            indices: [batch_size] dataset indices
            
        Returns:
            loss: scalar tensor
            rewards: [batch_size] tour lengths
        """
        if self.use_vmap:
            return self.compute_loss_vmap(nodes, indices)
        
        # Original implementation (fallback)
        batch_size = nodes.size(0)
        
        # Create environment and reset
        env = TSPEnv(nodes, device=nodes.device)
        obs = env.reset()
        
        log_probs = []
        
        # Rollout with gradients (training path - no caching)
        done = False
        while not done:
            # Get action from model (with gradients)
            action, log_prob = self.model_module.get_action(obs, deterministic=False)
            
            # Store log prob
            log_probs.append(log_prob)
            
            # Step environment
            obs, done = env.step(action)
        
        # Get rewards (tour lengths)
        rewards = env.get_tour_length()
        
        # Stack log probs
        log_probs = torch.stack(log_probs, dim=1)
        
        # Initialize moving average if needed
        if self.moving_avg is None:
            self.moving_avg = torch.zeros(len(indices), device=nodes.device)
        
        # Update baseline
        with torch.no_grad():
            self.moving_avg[indices] = moving_average(
                self.moving_avg[indices],
                rewards.detach(),
                self.beta
            )
        
        # Calculate advantage
        advantage = rewards - self.moving_avg[indices]
        
        # REINFORCE loss
        log_probs_sum = log_probs.sum(dim=1)
        log_probs_sum = log_probs_sum.clamp(min=-100)
        loss = (advantage * log_probs_sum).mean()
        
        return loss, rewards
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        train_loader.sampler.set_epoch(epoch)
        
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        
        iterator = tqdm(train_loader, desc=f"Epoch {epoch}") if self.rank == 0 else train_loader
        
        for batch_idx, (indices, batch) in enumerate(iterator):
            batch = batch.cuda(self.rank)
            batch_size = batch.size(0)
            
            # Forward pass with autocast
            with autocast('cuda'):
                loss, rewards = self.compute_loss(batch, indices)
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm(self.model.parameters(), self.args.GRAD_CLIP)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update meters
            loss_meter.update(loss.item(), batch_size)
            reward_meter.update(rewards.mean().item(), batch_size)
            
            # Periodic baseline sync
            self.step_count += 1
            if self.step_count % self.baseline_sync_freq == 0:
                self.sync_baseline()
        
        return loss_meter.avg, reward_meter.avg
    
    def evaluate(self, eval_loader, heuristic_distances=None):
        """Evaluate model performance using vmap."""
        if self.rank == 0:
            self.model.eval()
            
            all_rewards = []
            
            with torch.no_grad():
                for indices, batch in eval_loader:
                    batch = batch.cuda(self.rank)
                    
                    # Rollout with vmap
                    with autocast('cuda'):
                        rewards, _, _ = rollout_episode_vmap(
                            self.model_module, batch, device=batch.device
                        )
                    
                    all_rewards.append(rewards.cpu())
            
            all_rewards = torch.cat(all_rewards)
            avg_reward = all_rewards.mean().item()
            
            # Calculate gap if heuristic available
            gap = None
            if heuristic_distances is not None:
                ratio = all_rewards / heuristic_distances
                gap = ratio.mean().item()
            
            return avg_reward, gap
        else:
            return None, None
    
    def initialize_baseline(self, train_loader):
        """Initialize moving average baseline using vmap."""
        if self.rank == 0:
            print("Initializing baseline with vmap...")
        
        self.model.eval()
        
        if self.moving_avg is None:
            self.moving_avg = torch.zeros(len(train_loader.dataset), device=f'cuda:{self.rank}')
        
        with torch.no_grad():
            for indices, batch in train_loader:
                batch = batch.cuda(self.rank)
                
                with autocast('cuda'):
                    # Use vmap rollout
                    rewards, _, _ = rollout_episode_vmap(
                        self.model_module, batch, device=batch.device
                    )
                
                self.moving_avg[indices] = rewards
        
        # Sync baseline
        self.sync_baseline()
    
    def save_training_log(self, epoch, eval_reward, gap=None):
        """Save training log to checkpoint directory."""
        if self.rank == 0:
            log_path = os.path.join(self.checkpoint_dir, 'training_log.txt')
            with open(log_path, 'a') as f:
                f.write(f"Epoch {epoch}: Eval Reward = {eval_reward:.4f}")
                if gap is not None:
                    f.write(f", Gap = {gap:.4f}x")
                if epoch == self.best_epoch:
                    f.write(" [BEST]")
                f.write("\n")
    
    def train(self, train_loader, eval_loader, test_dataset=None):
        """Full training loop."""
        # Save configuration at the start
        self.save_config()
        
        # Compute heuristic solutions on rank 0
        heuristic_distances = None
        if self.rank == 0 and test_dataset is not None:
            print("Computing heuristic solutions...")
            heuristic_distances = []
            for i, (_, pointset) in enumerate(tqdm(test_dataset)):
                dist_val = get_heuristic_solution(pointset)
                if dist_val is not None:
                    heuristic_distances.append(dist_val)
                else:
                    heuristic_distances = None
                    break
            
            if heuristic_distances is not None:
                heuristic_distances = torch.tensor(heuristic_distances)
        
        # Initialize baseline
        self.initialize_baseline(train_loader)
        
        # Training loop
        for epoch in range(self.args.NUM_EPOCHS):
            # Train
            avg_loss, avg_reward = self.train_epoch(train_loader, epoch)
            
            # Evaluate
            eval_reward, gap = self.evaluate(eval_loader, heuristic_distances)
            
            # Print results and save best model
            if self.rank == 0:
                print(f"\n[Epoch {epoch}]")
                print(f"  Train Loss: {avg_loss:.4f}")
                print(f"  Train Reward: {avg_reward:.4f}")
                print(f"  Eval Reward: {eval_reward:.4f}")
                if gap is not None:
                    print(f"  Gap vs Heuristic: {gap:.4f}x")
                
                # Save best model (lower tour length is better)
                if eval_reward < self.best_eval_reward:
                    self.best_eval_reward = eval_reward
                    self.best_epoch = epoch
                    print(f"  New best model! Eval reward: {eval_reward:.4f}")
                    torch.save(self.model_module.state_dict(), "model.pth")
                    # Also save in checkpoint directory with epoch info
                    checkpoint_path = os.path.join(self.checkpoint_dir, f'best_model_epoch{epoch}.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model_module.state_dict(),
                        'eval_reward': eval_reward,
                        'gap': gap
                    }, checkpoint_path)
                    print(f"  Best model saved to model.pth and {checkpoint_path}")
                
                # Save training log
                self.save_training_log(epoch, eval_reward, gap)
        
        # Final summary
        if self.rank == 0:
            print(f"\n{'='*50}")
            print(f"Training completed!")
            print(f"Best model achieved at epoch {self.best_epoch}")
            print(f"Best eval reward: {self.best_eval_reward:.4f}")
            print(f"Model saved as 'model.pth'")
            print(f"Checkpoint and logs saved in '{self.checkpoint_dir}/' directory")
            print(f"{'='*50}")