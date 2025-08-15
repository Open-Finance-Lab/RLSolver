import torch
import copy
from scipy.stats import ttest_rel
from tqdm import tqdm
import torch.distributed as dist
from model import AttentionDynamicModel
from utils import generate_data_onfly_distributed


def rollout(model, dataset, batch_size=1000, device='cuda', disable_tqdm=False):
    """Evaluate model in greedy mode."""
    model.eval()
    model.set_decode_type("greedy")
    costs_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataset, disable=disable_tqdm, desc="Rollout"):
            batch = [b.to(device) for b in batch]
            cost, _ = model(batch)
            costs_list.append(cost)
    
    model.set_decode_type("sampling")
    return torch.cat(costs_list, dim=0)


def validate(dataset, model, batch_size=1000, device='cuda', rank=0):
    """Validate model on dataset."""
    model_to_eval = model.module if hasattr(model, 'module') else model
    val_costs = rollout(model_to_eval, dataset, batch_size, device, disable_tqdm=rank != 0)
    mean_cost = val_costs.mean()
    
    if dist.is_initialized():
        dist.all_reduce(mean_cost, op=dist.ReduceOp.SUM)
        mean_cost = mean_cost / dist.get_world_size()
    
    if rank == 0:
        print(f"Validation: {mean_cost.item():.4f}")
    
    return mean_cost


class RolloutBaseline:
    """Rollout baseline for REINFORCE algorithm with distributed support."""
    
    def __init__(self, model, config, epoch=0):
        self.config = config
        self.cur_epoch = epoch
        self.alpha = 0.0
        self.running_average_cost = None
        self.device = config.DEVICE
        
        self._update_baseline(model, epoch)
    
    def _update_baseline(self, model, epoch):
        """Update baseline model (synchronized across GPUs)."""
        self.model = copy.deepcopy(model)
        
        # Synchronize baseline model parameters across GPUs
        if dist.is_initialized():
            for param in self.model.parameters():
                dist.broadcast(param.data, src=0)
        
        # Generate baseline dataset
        self.dataset = generate_data_onfly_distributed(
            num_samples=self.config.BASELINE_SAMPLES,
            graph_size=self.config.GRAPH_SIZE,
            batch_size=self.config.BATCH_SIZE,
            rank=self.config.RANK,
            world_size=self.config.WORLD_SIZE,
            device=self.device
        )
        
        if self.config.RANK == 0:
            print(f"Evaluating baseline (epoch {epoch})")
        
        self.bl_vals = rollout(self.model, self.dataset, device=self.device, 
                              disable_tqdm=self.config.RANK != 0)
        self.mean = self.bl_vals.mean()
        
        if dist.is_initialized():
            dist.all_reduce(self.mean, op=dist.ReduceOp.SUM)
            self.mean = self.mean / dist.get_world_size()
        
        self.cur_epoch = epoch
    
    def ema_eval(self, cost):
        """Exponential moving average for warmup."""
        cost_mean = cost.mean()
        
        if dist.is_initialized():
            dist.all_reduce(cost_mean, op=dist.ReduceOp.SUM)
            cost_mean = cost_mean / dist.get_world_size()
        
        if self.running_average_cost is None:
            self.running_average_cost = cost_mean
        else:
            self.running_average_cost = (self.config.WARMUP_BETA * self.running_average_cost + 
                                        (1 - self.config.WARMUP_BETA) * cost_mean)
        return self.running_average_cost
    
    def eval(self, batch, cost):
        """Evaluate baseline value for batch."""
        if self.alpha == 0:
            return self.ema_eval(cost)
        
        if self.alpha < 1:
            v_ema = self.ema_eval(cost)
        else:
            v_ema = 0.0
        
        with torch.no_grad():
            self.model.eval()
            v_b, _ = self.model(batch)
            self.model.train()
        
        return self.alpha * v_b + (1 - self.alpha) * v_ema
    
    def eval_all(self, dataset):
        """Evaluate baseline on entire dataset."""
        if self.alpha < 1:
            return None
        return rollout(self.model, dataset, batch_size=2048, device=self.device,
                      disable_tqdm=self.config.RANK != 0)
    
    def epoch_callback(self, model, epoch):
        """Update baseline if candidate is better."""
        self.cur_epoch = epoch
        
        if self.config.RANK == 0:
            print(f"Comparing models (epoch {epoch})")
        
        candidate_vals = rollout(model, self.dataset, device=self.device,
                               disable_tqdm=self.config.RANK != 0)
        candidate_mean = candidate_vals.mean()
        
        if dist.is_initialized():
            dist.all_reduce(candidate_mean, op=dist.ReduceOp.SUM)
            candidate_mean = candidate_mean / dist.get_world_size()
        
        diff = candidate_mean - self.mean
        
        update_baseline = False
        if self.config.RANK == 0:
            status = "✓ Better" if diff < 0 else "✗ Worse"
            print(f"{status} | Candidate: {candidate_mean.item():.4f} | "
                  f"Baseline: {self.mean.item():.4f} | Δ: {diff.item():+.4f}")
            
            if diff < 0:
                t, p = ttest_rel(candidate_vals.cpu().numpy(), self.bl_vals.cpu().numpy())
                p_val = p / 2
                
                if p_val < 0.05:
                    print(f'→ Updating baseline (p={p_val:.4f})')
                    update_baseline = True
                else:
                    print(f'→ Not significant (p={p_val:.4f})')
        
        if dist.is_initialized():
            update_baseline = torch.tensor(update_baseline, dtype=torch.bool, device=self.device)
            dist.broadcast(update_baseline, src=0)
            update_baseline = update_baseline.item()
        
        if update_baseline:
            self._update_baseline(model, epoch)
        
        # Update warmup
        if self.alpha < 1.0:
            self.alpha = (epoch + 1) / float(self.config.WARMUP_EPOCHS)
            if self.config.RANK == 0:
                print(f"Warmup α: {self.alpha:.2f}")