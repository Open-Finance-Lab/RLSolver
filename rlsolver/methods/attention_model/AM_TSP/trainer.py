# trainer.py

import os
import json
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from datetime import datetime

from env import TSPEnv, VectorizedTSPEnv
from utils import clip_grad_norm, AverageMeter, get_heuristic_solution


@torch.compile(mode='reduce-overhead')
def rollout_episode_vmap_pomo_compiled(model, nodes, pomo_size=None, device='cuda'):
    """POMO rollout with vmap-style vectorization - optimized for zero-copy.
    
    Args:
        model: TSPActor model (unwrapped)
        nodes: [batch_size, seq_len, 2]
        pomo_size: Number of parallel rollouts (default: seq_len)
        
    Returns:
        tour_lengths: [batch_size, pomo_size]
        log_probs: [batch_size, pomo_size, seq_len-1]
        actions: [batch_size, pomo_size, seq_len]
    """
    batch_size = nodes.size(0)
    seq_len = nodes.size(1)
    if pomo_size is None:
        pomo_size = seq_len
    
    # Pre-compute encoder embeddings once per instance
    embedded = model.network.embedding(nodes)
    encoded = model.network.encoder(embedded)
    
    # Use view instead of expand + reshape to avoid copies
    # Expand creates a view, but reshape after expand might copy
    # Instead, we'll work with the expanded view directly
    encoded_expanded = encoded.unsqueeze(1).expand(batch_size, pomo_size, seq_len, -1)
    nodes_expanded = nodes.unsqueeze(1).expand(batch_size, pomo_size, seq_len, 2)
    
    # Create contiguous views only once
    encoded_flat = encoded_expanded.contiguous().view(batch_size * pomo_size, seq_len, -1)
    nodes_flat = nodes_expanded.contiguous().view(batch_size * pomo_size, seq_len, 2)
    
    # Initialize states
    visited_mask = torch.zeros(batch_size * pomo_size, seq_len, dtype=torch.bool, device=device)
    
    # Pre-create indices
    flat_indices = torch.arange(batch_size * pomo_size, device=device)
    pomo_indices = torch.arange(pomo_size, device=device).repeat(batch_size) % seq_len
    
    # POMO: Different starting nodes (no clone needed here)
    first_node = pomo_indices
    current_node = pomo_indices  # Direct assignment instead of clone
    visited_mask[flat_indices, current_node] = True
    
    # Pre-allocate tensors instead of using lists
    log_probs_tensor = torch.empty(batch_size * pomo_size, seq_len - 1, device=device)
    actions_tensor = torch.empty(batch_size * pomo_size, seq_len, dtype=torch.long, device=device)
    actions_tensor[:, 0] = current_node
    
    # Rollout loop
    for step in range(1, seq_len):
        obs = {
            'nodes': nodes_flat,
            'current_node': current_node,
            'first_node': first_node,
            'action_mask': ~visited_mask,
            'encoded': encoded_flat
        }
        
        action, log_prob = model.get_action(obs, deterministic=False)
        
        current_node = action
        visited_mask[flat_indices, action] = True
        
        # Direct tensor assignment instead of list append
        log_probs_tensor[:, step - 1] = log_prob
        actions_tensor[:, step] = action
    
    # Compute tour lengths
    vec_env = VectorizedTSPEnv(nodes_flat, device=device)
    tour_lengths = vec_env.compute_all_tours(actions_tensor)
    
    # Use view instead of reshape for final output
    tour_lengths = tour_lengths.view(batch_size, pomo_size)
    log_probs = log_probs_tensor.view(batch_size, pomo_size, seq_len - 1)
    actions = actions_tensor.view(batch_size, pomo_size, seq_len)
    
    return tour_lengths, log_probs, actions


# Wrapper function for evaluation (uses no_grad context)
def rollout_episode_vmap_pomo(model, nodes, pomo_size=None, device='cuda'):
    """POMO rollout wrapper that handles no_grad context."""
    with torch.no_grad():
        return rollout_episode_vmap_pomo_compiled(model, nodes, pomo_size, device)


class DistributedPOMOTrainer:
    """Distributed trainer with POMO and zero-copy optimizations."""
    
    def __init__(self, model, args, rank, world_size):
        self.model = model
        self.model_module = model.module if hasattr(model, 'module') else model
        self.args = args
        self.rank = rank
        self.world_size = world_size
        
        # POMO configuration - NOW USES NUM_TRAIN_ENVS
        self.pomo_size = args.NUM_TRAIN_ENVS
        
        # Adjust effective batch size for POMO
        self.effective_batch_size = args.BATCH_SIZE // self.pomo_size
        if self.effective_batch_size < 1:
            self.effective_batch_size = 1
            if self.rank == 0:
                print(f"Warning: Adjusting batch size to {self.effective_batch_size} due to POMO memory requirements")
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=args.LR)
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=args.NUM_EPOCHS,
            eta_min=args.LR * 0.01
        )
        
        # Mixed precision
        self.scaler = GradScaler()
        
        # Best model tracking
        self.best_eval_reward = float('inf')
        self.best_epoch = -1
        
        # Create checkpoint directory - NOW USES CONFIG PATH
        self.checkpoint_dir = args.CHECKPOINT_DIR
        if self.rank == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Compile the loss computation method
        self._compute_loss_compiled = torch.compile(
            self._compute_loss_core,
            mode='reduce-overhead'
        )
    
    def _compute_loss_core(self, nodes, model_module, pomo_size):
        """Core loss computation logic - optimized for zero-copy."""
        batch_size = nodes.size(0)
        seq_len = nodes.size(1)
        device = nodes.device
        
        # Encode only once per instance
        embedded = model_module.network.embedding(nodes)
        encoded = model_module.network.encoder(embedded)
        
        # Use view operations to avoid copies
        encoded_expanded = encoded.unsqueeze(1).expand(batch_size, pomo_size, seq_len, -1)
        nodes_expanded = nodes.unsqueeze(1).expand(batch_size, pomo_size, seq_len, 2)
        
        # Create contiguous views only once
        encoded_flat = encoded_expanded.contiguous().view(batch_size * pomo_size, seq_len, -1)
        nodes_flat = nodes_expanded.contiguous().view(batch_size * pomo_size, seq_len, 2)
        
        # Initialize states
        visited_mask = torch.zeros(batch_size * pomo_size, seq_len, dtype=torch.bool, device=device)
        
        # Pre-create indices
        flat_indices = torch.arange(batch_size * pomo_size, device=device)
        pomo_indices = torch.arange(pomo_size, device=device).repeat(batch_size) % seq_len
        
        # POMO: deterministic first action (no clone needed)
        first_node = pomo_indices
        current_node = pomo_indices  # Direct assignment
        visited_mask[flat_indices, current_node] = True
        
        # Pre-allocate tensors
        log_probs_tensor = torch.empty(batch_size * pomo_size, seq_len - 1, device=device)
        actions_tensor = torch.empty(batch_size * pomo_size, seq_len, dtype=torch.long, device=device)
        actions_tensor[:, 0] = current_node
        
        # Rollout with gradients
        for step in range(1, seq_len):
            obs = {
                'nodes': nodes_flat,
                'current_node': current_node,
                'first_node': first_node,
                'action_mask': ~visited_mask,
                'encoded': encoded_flat
            }
            
            action, log_prob = model_module.get_action(obs, deterministic=False)
            
            current_node = action
            visited_mask[flat_indices, action] = True
            
            # Direct assignment
            log_probs_tensor[:, step - 1] = log_prob
            actions_tensor[:, step] = action
        
        # Compute rewards
        vec_env = VectorizedTSPEnv(nodes_flat, device=device)
        rewards = vec_env.compute_all_tours(actions_tensor)
        
        # Use view for reshaping
        rewards = rewards.view(batch_size, pomo_size)
        log_probs = log_probs_tensor.view(batch_size, pomo_size, seq_len - 1)
        
        # POMO shared baseline
        baseline = rewards.mean(dim=1, keepdim=True)
        
        # Advantage
        advantage = rewards - baseline
        
        # Sum log probabilities
        log_probs_sum = log_probs.sum(dim=2)
        log_probs_sum = log_probs_sum.clamp(min=-100)
        
        # POMO loss
        loss = (advantage * log_probs_sum).mean()
        
        return loss, rewards
    
    def compute_loss_vmap_pomo(self, nodes):
        """Compute POMO loss with vmap-style vectorization - uses compiled version."""
        return self._compute_loss_compiled(nodes, self.model_module, self.pomo_size)
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch with POMO and zero-copy optimizations."""
        self.model.train()
        train_loader.sampler.set_epoch(epoch)
        
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        
        iterator = tqdm(train_loader, desc=f"Epoch {epoch} [LR={self.scheduler.get_last_lr()[0]:.6f}]") if self.rank == 0 else train_loader
        
        for batch_idx, (indices, batch) in enumerate(iterator):
            batch = batch.cuda(self.rank)
            batch_size = batch.size(0)
            
            # Forward pass with autocast
            with autocast('cuda'):
                loss, rewards = self.compute_loss_vmap_pomo(batch)
            
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
            reward_meter.update(rewards.min(dim=1)[0].mean().item(), batch_size)
        
        return loss_meter.avg, reward_meter.avg
    
    def evaluate(self, eval_loader, heuristic_distances=None):
        """Evaluate with POMO using zero-copy optimization."""
        if self.rank == 0:
            self.model.eval()
            
            # Pre-allocate list with known size for better memory efficiency
            num_batches = len(eval_loader)
            all_rewards = []
            
            with torch.no_grad():
                for indices, batch in eval_loader:
                    batch = batch.cuda(self.rank)
                    
                    # POMO rollout
                    rewards, _, _ = rollout_episode_vmap_pomo(
                        self.model_module, batch,
                        pomo_size=self.pomo_size,
                        device=batch.device
                    )
                    
                    # Take minimum tour length
                    best_rewards = rewards.min(dim=1)[0]
                    all_rewards.append(best_rewards)
            
            # Batch CPU transfer - more efficient than individual transfers
            all_rewards = torch.cat(all_rewards).cpu()
            avg_reward = all_rewards.mean().item()
            
            # Calculate gap if heuristic available
            gap = None
            if heuristic_distances is not None:
                ratio = all_rewards / heuristic_distances
                gap = ratio.mean().item()
            
            return avg_reward, gap
        else:
            return None, None
    
    def save_config(self):
        """Save training configuration."""
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
                    'algorithm': 'POMO with zero-copy optimizations + torch.compile',
                    'pomo_size': self.pomo_size,
                    'num_train_envs': self.args.NUM_TRAIN_ENVS,
                    'lr': self.args.LR,
                    'lr_min': self.args.LR * 0.01,
                    'num_epochs': self.args.NUM_EPOCHS,
                    'grad_clip': self.args.GRAD_CLIP,
                    'batch_size': self.args.BATCH_SIZE,
                    'effective_batch_size': self.effective_batch_size,
                    'compile_mode': 'reduce-overhead',
                    'optimizations': 'zero-copy memory operations',
                },
                'timestamp': datetime.now().isoformat(),
            }
            
            config_path = os.path.join(self.checkpoint_dir, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"Configuration saved to {config_path}")
    
    def save_checkpoint(self, epoch, eval_reward, gap=None):
        """Save model checkpoint."""
        if self.rank == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model_module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'eval_reward': eval_reward,
                'gap': gap,
                'best_eval_reward': self.best_eval_reward,
            }
            
            # Save as best model - NOW USES CONFIG PATH
            torch.save(self.model_module.state_dict(), self.args.MODEL_PATH)
            
            # Save full checkpoint
            checkpoint_path = os.path.join(self.checkpoint_dir, f'best_model_epoch{epoch}.pth')
            torch.save(checkpoint, checkpoint_path)
            
            return checkpoint_path
    
    def save_training_log(self, epoch, eval_reward, gap=None):
        """Save training log."""
        if self.rank == 0:
            log_path = os.path.join(self.checkpoint_dir, 'training_log.txt')
            with open(log_path, 'a') as f:
                f.write(f"Epoch {epoch}: Eval Reward = {eval_reward:.4f}")
                f.write(f", LR = {self.scheduler.get_last_lr()[0]:.6f}")
                if gap is not None:
                    f.write(f", Gap = {gap:.4f}x")
                if epoch == self.best_epoch:
                    f.write(" [BEST]")
                f.write("\n")
    
    def train(self, train_loader, eval_loader, test_dataset=None):
        """Full training loop with POMO and zero-copy optimizations."""
        # Save configuration
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
        
        if self.rank == 0:
            print(f"\nPOMO Training Configuration (with zero-copy optimizations):")
            print(f"  POMO size (NUM_TRAIN_ENVS): {self.pomo_size}")
            print(f"  Original batch size: {self.args.BATCH_SIZE}")
            print(f"  Effective batch size: {self.effective_batch_size}")
            print(f"  Total trajectories per step: {self.effective_batch_size * self.pomo_size * self.world_size}")
            print(f"  Compile mode: reduce-overhead")
            print(f"  Memory optimizations: zero-copy operations")
            print()
        
        # Training loop
        for epoch in range(self.args.NUM_EPOCHS):
            # Train
            avg_loss, avg_reward = self.train_epoch(train_loader, epoch)
            
            # Step learning rate scheduler
            self.scheduler.step()
            
            # Evaluate
            eval_reward, gap = self.evaluate(eval_loader, heuristic_distances)
            
            # Print results and save best model
            if self.rank == 0:
                print(f"\n[Epoch {epoch}]")
                print(f"  Train Loss: {avg_loss:.4f}")
                print(f"  Train Reward (best of POMO): {avg_reward:.4f}")
                print(f"  Eval Reward (best of POMO): {eval_reward:.4f}")
                print(f"  Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
                if gap is not None:
                    print(f"  Gap vs Heuristic: {gap:.4f}x")
                
                # Save best model
                if eval_reward < self.best_eval_reward:
                    self.best_eval_reward = eval_reward
                    self.best_epoch = epoch
                    print(f"  New best model! Eval reward: {eval_reward:.4f}")
                    checkpoint_path = self.save_checkpoint(epoch, eval_reward, gap)
                    print(f"  Model saved to {self.args.MODEL_PATH} and {checkpoint_path}")
                
                # Save training log
                self.save_training_log(epoch, eval_reward, gap)
        
        # Final summary
        if self.rank == 0:
            print(f"\n{'='*50}")
            print(f"Training completed with POMO + zero-copy optimizations!")
            print(f"Best model achieved at epoch {self.best_epoch}")
            print(f"Best eval reward: {self.best_eval_reward:.4f}")
            print(f"Model saved as '{self.args.MODEL_PATH}'")
            print(f"Checkpoint and logs saved in '{self.checkpoint_dir}/' directory")
            print(f"{'='*50}")