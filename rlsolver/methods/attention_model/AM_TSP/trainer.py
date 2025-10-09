"""Distributed POMO Trainer using TSPEnv."""

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
from util import clip_grad_norm, AverageMeter, get_heuristic_solution


class DistributedPOMOTrainer:
    """Distributed trainer with POMO using TSPEnv."""
    
    def __init__(self, model, args, rank, world_size):
        self.model = model
        self.model_module = model.module if hasattr(model, 'module') else model
        self.args = args
        self.rank = rank
        self.world_size = world_size
        
        self.pomo_size = args.NUM_TRAIN_ENVS
        
        self.effective_batch_size = args.BATCH_SIZE // self.pomo_size
        if self.effective_batch_size < 1:
            self.effective_batch_size = 1
            if self.rank == 0:
                print(f"Warning: Adjusting batch size to {self.effective_batch_size} due to POMO memory requirements")
        
        self.optimizer = optim.Adam(model.parameters(), lr=args.LR)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=args.NUM_EPOCHS,
            eta_min=args.LR * 0.01
        )
        
        self.scaler = GradScaler()
        
        self.best_eval_reward = float('inf')
        self.best_epoch = -1
        
        self.checkpoint_dir = args.CHECKPOINT_DIR
        if self.rank == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def compute_loss_pomo(self, nodes):
        """Compute POMO loss using TSPEnv.
        
        Args:
            nodes: [batch_size, seq_len, 2]
        Returns:
            loss: scalar
            rewards: [batch_size, pomo_size]
        """
        env = TSPEnv(nodes, device=nodes.device)
        
        # POMO rollout
        rewards, log_probs, _ = env.rollout_pomo(
            self.model_module,
            pomo_size=self.pomo_size
        )
        
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
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        train_loader.sampler.set_epoch(epoch)
        
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        
        iterator = tqdm(
            train_loader,
            desc=f"Epoch {epoch} [LR={self.scheduler.get_last_lr()[0]:.6f}]"
        ) if self.rank == 0 else train_loader
        
        for batch_idx, (indices, batch) in enumerate(iterator):
            batch = batch.cuda(self.rank)
            batch_size = batch.size(0)
            
            with autocast('cuda'):
                loss, rewards = self.compute_loss_pomo(batch)
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm(self.model.parameters(), self.args.GRAD_CLIP)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            loss_meter.update(loss.item(), batch_size)
            reward_meter.update(rewards.min(dim=1)[0].mean().item(), batch_size)
        
        return loss_meter.avg, reward_meter.avg
    
    def evaluate(self, eval_loader, heuristic_distances=None):
        """Evaluate with POMO."""
        if self.rank == 0:
            self.model.eval()
            
            all_rewards = []
            
            with torch.no_grad():
                for indices, batch in eval_loader:
                    batch = batch.cuda(self.rank)
                    
                    env = TSPEnv(batch, device=batch.device)
                    rewards, _, _ = env.rollout_pomo(
                        self.model_module,
                        pomo_size=self.pomo_size
                    )
                    
                    best_rewards = rewards.min(dim=1)[0]
                    all_rewards.append(best_rewards)
            
            all_rewards = torch.cat(all_rewards).cpu()
            avg_reward = all_rewards.mean().item()
            
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
                    'algorithm': 'POMO with zero-copy optimizations',
                    'pomo_size': self.pomo_size,
                    'num_train_envs': self.args.NUM_TRAIN_ENVS,
                    'lr': self.args.LR,
                    'lr_min': self.args.LR * 0.01,
                    'num_epochs': self.args.NUM_EPOCHS,
                    'grad_clip': self.args.GRAD_CLIP,
                    'batch_size': self.args.BATCH_SIZE,
                    'effective_batch_size': self.effective_batch_size,
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
            
            torch.save(self.model_module.state_dict(), self.args.MODEL_PATH)
            
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f'best_model_epoch{epoch}.pth'
            )
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
        """Full training loop."""
        self.save_config()
        
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
            print(f"\nPOMO Training Configuration:")
            print(f" POMO size (NUM_TRAIN_ENVS): {self.pomo_size}")
            print(f" Original batch size: {self.args.BATCH_SIZE}")
            print(f" Effective batch size: {self.effective_batch_size}")
            print(f" Total trajectories per step: {self.effective_batch_size * self.pomo_size * self.world_size}")
            print(f" Memory optimizations: zero-copy operations")
            print()
        
        for epoch in range(self.args.NUM_EPOCHS):
            avg_loss, avg_reward = self.train_epoch(train_loader, epoch)
            self.scheduler.step()
            
            eval_reward, gap = self.evaluate(eval_loader, heuristic_distances)
            
            if self.rank == 0:
                print(f"\n[Epoch {epoch}]")
                print(f" Train Loss: {avg_loss:.4f}")
                print(f" Train Reward (best of POMO): {avg_reward:.4f}")
                print(f" Eval Reward (best of POMO): {eval_reward:.4f}")
                print(f" Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
                if gap is not None:
                    print(f" Gap vs Heuristic: {gap:.4f}x")
                
                if eval_reward < self.best_eval_reward:
                    self.best_eval_reward = eval_reward
                    self.best_epoch = epoch
                    print(f" New best model! Eval reward: {eval_reward:.4f}")
                    checkpoint_path = self.save_checkpoint(epoch, eval_reward, gap)
                    print(f" Model saved to {self.args.MODEL_PATH} and {checkpoint_path}")
                
                self.save_training_log(epoch, eval_reward, gap)
        
        if self.rank == 0:
            print(f"\n{'='*50}")
            print(f"Training completed!")
            print(f"Best model achieved at epoch {self.best_epoch}")
            print(f"Best eval reward: {self.best_eval_reward:.4f}")
            print(f"Model saved as '{self.args.MODEL_PATH}'")
            print(f"Checkpoint and logs saved in '{self.checkpoint_dir}/' directory")
            print(f"{'='*50}")