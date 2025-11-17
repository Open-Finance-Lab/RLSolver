# trainer.py

import os
import json
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
from datetime import datetime

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
    batch_idx = torch.arange(batch_size, device=nodes.device)[:, None, None].expand(-1, pomo_size, seq_len)
    tour_nodes = nodes[batch_idx, actions]
    diffs = tour_nodes[:, :, 1:] - tour_nodes[:, :, :-1]
    distances = torch.norm(diffs, dim=3)
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

    embedded = model.network.embedding(nodes)
    encoded = model.network.encoder(embedded)

    visited_mask = torch.zeros(batch_size, pomo_size, seq_len, dtype=torch.bool, device=device)
    
    pomo_indices = torch.arange(pomo_size, device=device) % seq_len
    first_node = pomo_indices.unsqueeze(0).expand(batch_size, -1)
    current_node = first_node.clone()

    batch_idx = torch.arange(batch_size, device=device)[:, None].expand(-1, pomo_size)
    pomo_idx = torch.arange(pomo_size, device=device)[None, :].expand(batch_size, -1)
    visited_mask[batch_idx, pomo_idx, current_node] = True

    log_probs_list = []
    actions_list = [current_node]

    for step in range(1, seq_len):
        obs = {
            'nodes': nodes,
            'current_node': current_node,
            'first_node': first_node,
            'action_mask': ~visited_mask,
            'encoded': encoded
        }
        action, log_prob = model.get_action(obs, deterministic=False)
        current_node = action
        visited_mask[batch_idx, pomo_idx, action] = True
        log_probs_list.append(log_prob)
        actions_list.append(action)

    log_probs = torch.stack(log_probs_list, dim=2)
    actions = torch.stack(actions_list, dim=2)
    tour_lengths = compute_tour_lengths_structured(nodes, actions)
    
    return tour_lengths, log_probs, actions


def rollout_episode_pomo_structured(model, nodes, pomo_size=None, device='cuda'):
    """POMO rollout wrapper that handles no_grad context."""
    with torch.no_grad():
        return rollout_episode_pomo_structured_compiled(model, nodes, pomo_size, device)


class DistributedPOMOTrainer:
    """Distributed trainer with memory-efficient POMO and gradient checkpointing."""
    def __init__(self, model, args, rank, world_size):
        self.model = model
        self.model_module = model.module if hasattr(model, 'module') else model
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.gpu_id = args.TRAIN_GPU_IDS[rank]
        self.pomo_size = args.NUM_POMO
        self.num_envs_per_gpu = args.NUM_ENVS // world_size
        self.batch_size = args.BATCH_SIZE
        self.accumulation_steps = (self.num_envs_per_gpu + self.batch_size - 1) // self.batch_size
        
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

    def _encoder_forward(self, embedded, encoder):
        """Wrapper for encoder forward pass (for gradient checkpointing)."""
        return encoder(embedded)

    def _decoder_step(self, model_module, nodes, current_node, first_node, action_mask, encoded):
        """Wrapper for single decoder step (for gradient checkpointing)."""
        obs = {
            'nodes': nodes,
            'current_node': current_node,
            'first_node': first_node,
            'action_mask': action_mask,
            'encoded': encoded
        }
        return model_module.get_action(obs, deterministic=False)

    def _compute_loss_core(self, nodes, model_module, pomo_size):
        """Core loss computation with gradient checkpointing."""
        batch_size = nodes.size(0)
        seq_len = nodes.size(1)
        device = nodes.device

        embedded = model_module.network.embedding(nodes)
        encoded = checkpoint(
            self._encoder_forward,
            embedded,
            model_module.network.encoder,
            use_reentrant=False
        )

        visited_mask = torch.zeros(batch_size, pomo_size, seq_len, dtype=torch.bool, device=device)
        
        pomo_indices = torch.arange(pomo_size, device=device) % seq_len
        first_node = pomo_indices.unsqueeze(0).expand(batch_size, -1)
        current_node = first_node.clone()

        batch_idx = torch.arange(batch_size, device=device)[:, None].expand(-1, pomo_size)
        pomo_idx = torch.arange(pomo_size, device=device)[None, :].expand(batch_size, -1)
        visited_mask[batch_idx, pomo_idx, current_node] = True

        log_probs_list = []
        actions_list = [current_node]

        for step in range(1, seq_len):
            action_mask = ~visited_mask
            action, log_prob = checkpoint(
                self._decoder_step,
                model_module,
                nodes,
                current_node,
                first_node,
                action_mask,
                encoded,
                use_reentrant=False
            )
            current_node = action
            visited_mask[batch_idx, pomo_idx, action] = True
            log_probs_list.append(log_prob)
            actions_list.append(action)

        log_probs = torch.stack(log_probs_list, dim=2)
        actions = torch.stack(actions_list, dim=2)
        rewards = compute_tour_lengths_structured(nodes, actions)
        
        baseline = rewards.mean(dim=1, keepdim=True)
        advantage = rewards - baseline
        log_probs_sum = log_probs.sum(dim=2).clamp(min=-5.0 * seq_len)
        loss = (advantage * log_probs_sum).mean()
        
        return loss, rewards

    def compute_loss_vmap_pomo(self, nodes):
        """Compute POMO loss with gradient checkpointing."""
        return self._compute_loss_core(nodes, self.model_module, self.pomo_size)

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch with gradient accumulation."""
        self.model.train()
        train_loader.sampler.set_epoch(epoch)
        
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        
        iterator = tqdm(train_loader, desc=f"Epoch {epoch} [LR={self.scheduler.get_last_lr()[0]:.6f}]") if self.rank == 0 else train_loader
        
        for batch_idx, (indices, batch) in enumerate(iterator):
            batch = batch.cuda(self.gpu_id)
            num_envs = batch.size(0)
            
            self.optimizer.zero_grad()
            
            # Split into micro-batches for gradient accumulation
            total_loss = 0.0
            total_rewards = []
            
            for start_idx in range(0, num_envs, self.batch_size):
                end_idx = min(start_idx + self.batch_size, num_envs)
                micro_batch = batch[start_idx:end_idx]
                micro_batch_size = micro_batch.size(0)
                
                with autocast('cuda'):
                    loss, rewards = self.compute_loss_vmap_pomo(micro_batch)
                    # Scale loss by the proportion of this micro-batch
                    loss = loss * (micro_batch_size / num_envs)
                
                self.scaler.scale(loss).backward()
                total_loss += loss.item() * (num_envs / micro_batch_size)
                total_rewards.append(rewards)
            
            # Gradient clipping and optimization step
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm(self.model.parameters(), self.args.GRAD_CLIP)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Aggregate metrics
            all_rewards = torch.cat(total_rewards, dim=0)
            loss_meter.update(total_loss, num_envs)
            reward_meter.update(all_rewards.min(dim=1)[0].mean().item(), num_envs)

        return loss_meter.avg, reward_meter.avg

    def evaluate(self, eval_loader, heuristic_distances=None):
        """Evaluate with memory-efficient POMO."""
        if self.rank == 0:
            self.model.eval()
            all_rewards = []
            
            with torch.no_grad():
                for indices, batch in eval_loader:
                    batch = batch.cuda(self.gpu_id)
                    rewards, _, _ = rollout_episode_pomo_structured(
                        self.model_module, batch,
                        pomo_size=self.pomo_size,
                        device=batch.device
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
                    'algorithm': 'POMO with gradient checkpointing and accumulation',
                    'pomo_size': self.pomo_size,
                    'num_envs': self.args.NUM_ENVS,
                    'batch_size': self.batch_size,
                    'num_envs_per_gpu': self.num_envs_per_gpu,
                    'accumulation_steps': self.accumulation_steps,
                    'lr': self.args.LR,
                    'lr_min': self.args.LR * 0.01,
                    'num_epochs': self.args.NUM_EPOCHS,
                    'grad_clip': self.args.GRAD_CLIP,
                    'world_size': self.world_size,
                    'memory_optimization': 'gradient checkpointing + gradient accumulation',
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
        """Full training loop with gradient checkpointing."""
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
            print(f"  POMO size: {self.pomo_size}")
            print(f"  Total NUM_ENVS: {self.args.NUM_ENVS}")
            print(f"  NUM_ENVS per GPU: {self.num_envs_per_gpu}")
            print(f"  Gradient batch size: {self.batch_size}")
            print()

        for epoch in range(self.args.NUM_EPOCHS):
            avg_loss, avg_reward = self.train_epoch(train_loader, epoch)
            self.scheduler.step()
            eval_reward, gap = self.evaluate(eval_loader, heuristic_distances)

            if self.rank == 0:
                print(f"\n[Epoch {epoch}]")
                print(f"  Train Loss: {avg_loss:.4f}")
                print(f"  Train Reward (best of POMO): {avg_reward:.4f}")
                print(f"  Eval Reward (best of POMO): {eval_reward:.4f}")
                print(f"  Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
                if gap is not None:
                    print(f"  Gap vs Heuristic: {gap:.4f}x")

                if eval_reward < self.best_eval_reward:
                    self.best_eval_reward = eval_reward
                    self.best_epoch = epoch
                    print(f"  New best model! Eval reward: {eval_reward:.4f}")
                    checkpoint_path = self.save_checkpoint(epoch, eval_reward, gap)
                    print(f"  Model saved to {self.args.MODEL_PATH} and {checkpoint_path}")

                self.save_training_log(epoch, eval_reward, gap)

        if self.rank == 0:
            print(f"\n{'='*50}")
            print(f"Training completed with gradient checkpointing!")
            print(f"Best model achieved at epoch {self.best_epoch}")
            print(f"Best eval reward: {self.best_eval_reward:.4f}")
            print(f"Model saved as '{self.args.MODEL_PATH}'")
            print(f"Checkpoint and logs saved in '{self.checkpoint_dir}/' directory")
            print(f"{'='*50}")
