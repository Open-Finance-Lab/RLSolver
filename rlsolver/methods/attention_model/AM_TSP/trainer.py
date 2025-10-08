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

from env import TSPEnv
from utils import clip_grad_norm, AverageMeter, get_heuristic_solution


def compute_tour_lengths_structured(nodes, actions):
    """Compute tour lengths without expanding nodes.
    
    Args:
        nodes: [batch_size, seq_len, 2]
        actions: [batch_size, pomo_size, seq_len]
    
    Returns:
        lengths: [batch_size, pomo_size]
    """
    batch_size, pomo_size, seq_len = actions.shape
    
    # Use advanced indexing to gather tour nodes
    batch_idx = torch.arange(batch_size, device=nodes.device)[:, None, None].expand(-1, pomo_size, seq_len)
    tour_nodes = nodes[batch_idx, actions]  # [batch, pomo, seq_len, 2]
    
    # Calculate distances
    diffs = tour_nodes[:, :, 1:] - tour_nodes[:, :, :-1]
    distances = torch.norm(diffs, dim=3)
    
    # Add distance from last to first
    last_to_first = torch.norm(tour_nodes[:, :, -1] - tour_nodes[:, :, 0], dim=2)
    
    lengths = distances.sum(dim=2) + last_to_first
    return lengths


@torch.compile(mode='reduce-overhead')
def rollout_episode_pomo_structured_compiled(model, nodes, pomo_size=None, device='cuda'):
    """POMO rollout with structured batching (no physical expansion).
    
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
    
    # Pre-compute encoder embeddings ONCE (shared across all POMO)
    embedded = model.network.embedding(nodes)
    encoded = model.network.encoder(embedded)  # [batch, seq_len, embed_dim]
    
    # Initialize states - structured format
    visited_mask = torch.zeros(batch_size, pomo_size, seq_len, dtype=torch.bool, device=device)
    
    # POMO: Different starting nodes for each rollout
    pomo_indices = torch.arange(pomo_size, device=device) % seq_len
    first_node = pomo_indices.unsqueeze(0).expand(batch_size, -1)  # [batch, pomo]
    current_node = first_node.clone()
    
    # Mark first nodes as visited using advanced indexing
    batch_idx = torch.arange(batch_size, device=device)[:, None].expand(-1, pomo_size)
    pomo_idx = torch.arange(pomo_size, device=device)[None, :].expand(batch_size, -1)
    visited_mask[batch_idx, pomo_idx, current_node] = True
    
    # Pre-allocate results
    log_probs_list = []
    actions_list = [current_node]
    
    # Rollout loop
    for step in range(1, seq_len):
        obs = {
            'nodes': nodes,  # Shared, not expanded
            'current_node': current_node,  # [batch, pomo]
            'first_node': first_node,  # [batch, pomo]
            'action_mask': ~visited_mask,  # [batch, pomo, seq_len]
            'encoded': encoded  # Shared encoding
        }
        
        action, log_prob = model.get_action(obs, deterministic=False)
        # action: [batch, pomo], log_prob: [batch, pomo]
        
        current_node = action
        visited_mask[batch_idx, pomo_idx, action] = True
        
        log_probs_list.append(log_prob)
        actions_list.append(action)
    
    # Stack results
    log_probs = torch.stack(log_probs_list, dim=2)  # [batch, pomo, seq_len-1]
    actions = torch.stack(actions_list, dim=2)  # [batch, pomo, seq_len]
    
    # Compute tour lengths
    tour_lengths = compute_tour_lengths_structured(nodes, actions)
    
    return tour_lengths, log_probs, actions


def rollout_episode_pomo_structured(model, nodes, pomo_size=None, device='cuda'):
    """POMO rollout wrapper that handles no_grad context."""
    with torch.no_grad():
        return rollout_episode_pomo_structured_compiled(model, nodes, pomo_size, device)


# Keep the old function name for compatibility but use new implementation
rollout_episode_vmap_pomo = rollout_episode_pomo_structured


class DistributedPOMOTrainer:
    """Distributed trainer with memory-efficient POMO."""
    
    def __init__(self, model, args, rank, world_size):
        self.model = model
        self.model_module = model.module if hasattr(model, 'module') else model
        self.args = args
        self.rank = rank
        self.world_size = world_size
        
        # POMO configuration
        self.pomo_size = args.NUM_TRAIN_ENVS
        
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
        
        # Create checkpoint directory
        self.checkpoint_dir = args.CHECKPOINT_DIR
        if self.rank == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Compile the optimized loss computation
        self._compute_loss_compiled = torch.compile(
            self._compute_loss_core,
            mode='reduce-overhead'
        )
    
    def _compute_loss_core(self, nodes, model_module, pomo_size):
        """Core loss computation with structured batching."""
        batch_size = nodes.size(0)
        seq_len = nodes.size(1)
        device = nodes.device
        
        # Encode ONCE (shared across all POMO)
        embedded = model_module.network.embedding(nodes)
        encoded = model_module.network.encoder(embedded)  # [batch, seq_len, embed_dim]
        
        # Initialize structured states
        visited_mask = torch.zeros(batch_size, pomo_size, seq_len, dtype=torch.bool, device=device)
        
        # POMO starting nodes
        pomo_indices = torch.arange(pomo_size, device=device) % seq_len
        first_node = pomo_indices.unsqueeze(0).expand(batch_size, -1)  # [batch, pomo]
        current_node = first_node.clone()
        
        # Mark first nodes as visited
        batch_idx = torch.arange(batch_size, device=device)[:, None].expand(-1, pomo_size)
        pomo_idx = torch.arange(pomo_size, device=device)[None, :].expand(batch_size, -1)
        visited_mask[batch_idx, pomo_idx, current_node] = True
        
        # Collect log probabilities and actions
        log_probs_list = []
        actions_list = [current_node]
        
        # Rollout with gradients
        for step in range(1, seq_len):
            obs = {
                'nodes': nodes,  # Shared, not expanded
                'current_node': current_node,  # [batch, pomo]
                'first_node': first_node,  # [batch, pomo]
                'action_mask': ~visited_mask,  # [batch, pomo, seq_len]
                'encoded': encoded  # Shared encoding
            }
            
            action, log_prob = model_module.get_action(obs, deterministic=False)
            
            current_node = action
            visited_mask[batch_idx, pomo_idx, action] = True
            
            log_probs_list.append(log_prob)
            actions_list.append(action)
        
        # Stack results
        log_probs = torch.stack(log_probs_list, dim=2)  # [batch, pomo, seq_len-1]
        actions = torch.stack(actions_list, dim=2)  # [batch, pomo, seq_len]
        
        # Compute rewards (tour lengths)
        rewards = compute_tour_lengths_structured(nodes, actions)  # [batch, pomo]
        
        # POMO shared baseline
        baseline = rewards.mean(dim=1, keepdim=True)
        advantage = rewards - baseline
        
        # Sum log probabilities
        log_probs_sum = log_probs.sum(dim=2).clamp(min=-5.0 * seq_len)
        
        # POMO loss
        loss = (advantage * log_probs_sum).mean()
        
        return loss, rewards
    
    def compute_loss_vmap_pomo(self, nodes):
        """Compute POMO loss with memory-efficient structured batching."""
        return self._compute_loss_compiled(nodes, self.model_module, self.pomo_size)
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch with memory-efficient POMO."""
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
        """Evaluate with memory-efficient POMO."""
        if self.rank == 0:
            self.model.eval()
            all_rewards = []
            
            with torch.no_grad():
                for indices, batch in eval_loader:
                    batch = batch.cuda(self.rank)
                    
                    # POMO rollout with structured batching
                    rewards, _, _ = rollout_episode_pomo_structured(
                        self.model_module, batch,
                        pomo_size=self.pomo_size,
                        device=batch.device
                    )
                    
                    # Take minimum tour length
                    best_rewards = rewards.min(dim=1)[0]
                    all_rewards.append(best_rewards)
            
            # Batch CPU transfer
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
                    'algorithm': 'POMO with memory-efficient structured batching',
                    'pomo_size': self.pomo_size,
                    'num_train_envs': self.args.NUM_TRAIN_ENVS,
                    'lr': self.args.LR,
                    'lr_min': self.args.LR * 0.01,
                    'num_epochs': self.args.NUM_EPOCHS,
                    'grad_clip': self.args.GRAD_CLIP,
                    'batch_size': self.args.BATCH_SIZE,
                    'batch_size_per_gpu': self.args.BATCH_SIZE // self.world_size,
                    'world_size': self.world_size,
                    'compile_mode': 'reduce-overhead',
                    'optimizations': 'structured batching, shared encoding, zero-copy operations',
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
            
            # Save as best model
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
        """Full training loop with memory-efficient POMO."""
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
            batch_size_per_gpu = self.args.BATCH_SIZE // self.world_size
            print(f"\nPOMO Training Configuration (Memory-Efficient):")
            print(f"  POMO size: {self.pomo_size}")
            print(f"  Total batch size: {self.args.BATCH_SIZE}")
            print(f"  Batch size per GPU: {batch_size_per_gpu}")
            print(f"  TSP instances per GPU: {batch_size_per_gpu}")
            print(f"  Trajectories per GPU: {batch_size_per_gpu * self.pomo_size}")
            print(f"  Total trajectories per step: {self.args.BATCH_SIZE * self.pomo_size}")
            print(f"  Memory optimization: Structured batching with shared encoding")
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
            print(f"Training completed with memory-efficient POMO!")
            print(f"Best model achieved at epoch {self.best_epoch}")
            print(f"Best eval reward: {self.best_eval_reward:.4f}")
            print(f"Model saved as '{self.args.MODEL_PATH}'")
            print(f"Checkpoint and logs saved in '{self.checkpoint_dir}/' directory")
            print(f"{'='*50}")