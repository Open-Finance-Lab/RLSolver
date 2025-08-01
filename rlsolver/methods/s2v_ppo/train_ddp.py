# train_ddp.py
import os
import sys
import torch
import torch.distributed as dist
import torch.nn.parallel
import numpy as np
from collections import deque
import random
from torch_scatter import scatter_add
from tqdm import tqdm
from torch_geometric.data import Data, Batch

from models import PPOLinearModel
from ppo_trainer import PPOTrainer
from data_utils import generate_batch_graphs, generate_graph
from config import Config
from env import MaxCutEnv


def main_worker(rank, world_size):
    # 初始化进程组
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    
    # 配置设置
    config = Config()
    config.device = f"cuda:{rank}"
    
    # 性能设置
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    
    # 设置随机种子 - 修复：所有进程使用相同的种子初始化模型
    if config.seed is not None:
        torch.manual_seed(config.seed)  # 移除 + rank
        np.random.seed(config.seed + rank)  # 数据生成仍可以不同
        random.seed(config.seed + rank)     # 数据生成仍可以不同
        torch.cuda.manual_seed(config.seed)  # 移除 + rank
    
    # 确保能整除
    assert config.num_parallel_envs % world_size == 0, f"num_parallel_envs must be divisible by world_size"
    assert config.batch_size % world_size == 0, f"batch_size must be divisible by world_size"
    
    # 每个进程的本地批次大小和环境数
    local_batch_size = config.batch_size // world_size
    local_num_envs = config.num_parallel_envs // world_size
    
    # 创建模型并移动到对应GPU
    model = PPOLinearModel(config).to(config.device)
    
    # 包装为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(
        model, 
        device_ids=[rank], 
        find_unused_parameters=False
    )
    
    if rank == 0:
        print(f"开始DDP训练，总GPU数: {world_size}, 设备: {config.device}")
        print(f"模型参数: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
        print(f"每个进程处理环境数: {local_num_envs}")
        print(f"每个进程批次大小: {local_batch_size}")
    
    trainer = PPOTrainer(model, config)
    
    # 只在主进程显示进度条
    if rank == 0:
        progress_bar = tqdm(range(config.epochs), desc="训练进度")
    else:
        progress_bar = range(config.epochs)
    
    # 每个环境的步数
    steps_per_env = local_batch_size // local_num_envs
    
    for epoch in progress_bar:
        model.train()
        
        # 初始化环境
        envs = []
        graph_datas = generate_batch_graphs(local_num_envs)
        
        for i in range(local_num_envs):
            env = MaxCutEnv(graph_datas[i], config)
            envs.append(env)
        
        # 批量重置所有环境
        states = [env.reset() for env in envs]
        
        # 创建批量数据
        data_list = [Data(x=state['x'], edge_index=state['edge_index']) for state in states]
        
        # 为每个环境创建独立的轨迹列表
        trajectories = [[] for _ in range(local_num_envs)]
        
        total_reward = 0.0
        total_steps = 0
        best_cuts = []
        
        with torch.no_grad():
            for step in range(steps_per_env):
                # 批量创建所有环境的数据
                batch = Batch.from_data_list(data_list).to(config.device)
                
                # 合并所有环境的mask
                batch_masks = torch.cat([states[i]['valid_actions_mask'] for i in range(local_num_envs)])
                
                # 批量前向传播
                logits, values = model(
                    batch.x,
                    batch.edge_index,
                    batch_masks,
                    batch.batch
                )
                
                # 批量采样动作
                ptr = batch.ptr
                env_logits = [logits[ptr[i]:ptr[i+1]] for i in range(local_num_envs)]
                actions = []
                log_probs = []
                
                for i in range(local_num_envs):
                    action_dist = torch.distributions.Categorical(logits=env_logits[i])
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action)
                    actions.append(action.item())
                    log_probs.append(log_prob.item())
                
                # 批量执行环境步骤
                for i in range(local_num_envs):
                    next_state, reward, done, info = envs[i].step(actions[i])
                    
                    # 添加到轨迹 - 修复：正确提取value
                    trajectories[i].append({
                        'x': states[i]['x'].clone(),
                        'edge_index': states[i]['edge_index'],
                        'valid_actions_mask': states[i]['valid_actions_mask'].clone(),
                        'action': actions[i],
                        'log_prob': log_probs[i],
                        'value': values[i].item(),  # 修复：直接使用第i个图的价值
                        'reward': reward,
                        'done': done
                    })
                    
                    total_reward += reward
                    total_steps += 1
                    
                    if done:
                        best_cuts.append(info['best_cut'])
                        # 生成新图并重置
                        new_graph_data = generate_graph(np.random.choice(['BA', 'ER', 'PL']))
                        envs[i] = MaxCutEnv(new_graph_data, config)
                        states[i] = envs[i].reset()
                        data_list[i] = Data(x=states[i]['x'], edge_index=states[i]['edge_index'])
                    else:
                        states[i] = next_state
                        data_list[i].x = next_state['x']
        
        # 处理截断情况 - 为未结束的轨迹计算最终价值
        with torch.no_grad():
            for i in range(local_num_envs):
                if trajectories[i] and not trajectories[i][-1]['done']:
                    # 批量计算最终价值
                    final_batch = Batch.from_data_list([data_list[i]]).to(config.device)
                    final_mask = states[i]['valid_actions_mask']
                    
                    _, final_value = model(
                        final_batch.x,
                        final_batch.edge_index,
                        final_mask,
                        final_batch.batch
                    )
                    
                    # 修复：正确提取final_value
                    trajectories[i].append({
                        'final_value': final_value[0].item()  # 修复：final_value形状为(1,)
                    })
        
        # 格式化轨迹 - 现在每个轨迹都是时序连续的
        formatted_trajectories = [
            {'steps': traj, 'final_cut': 0} 
            for traj in trajectories 
            if traj
        ]
        
        # PPO更新
        if formatted_trajectories:
            loss = trainer.update(formatted_trajectories)
        else:
            loss = 0.0
        
        # 创建包含总奖励、总步数和损失的张量
        local_stats = torch.tensor(
            [total_reward, total_steps, loss], 
            dtype=torch.float64, 
            device=config.device
        )
        
        # 计算本地统计
        avg_cut = np.mean(best_cuts) if best_cuts else 0.0
        local_episodes = len(best_cuts)
        
        # 创建包含平均切割值和回合数的张量
        local_metrics = torch.tensor(
            [avg_cut, local_episodes], 
            dtype=torch.float64, 
            device=config.device
        )
        
        # 同步所有进程的结果
        dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_metrics, op=dist.ReduceOp.SUM)
        
        # 计算全局统计
        global_total_reward = local_stats[0].item()
        global_total_steps = local_stats[1].item()
        global_total_loss = local_stats[2].item()
        
        # 正确计算全局平均值
        avg_reward_global = global_total_reward / max(global_total_steps, 1)
        avg_loss_global = global_total_loss / world_size
        avg_cut_global = local_metrics[0].item() / world_size
        num_episodes = int(local_metrics[1].item())
        
        if rank == 0:
            progress_bar.set_postfix({
                'loss': f'{avg_loss_global:.4f}',
                'avg_reward': f'{avg_reward_global:.4f}',
                'avg_cut': f'{avg_cut_global:.2f}',
                'lr': f'{trainer.scheduler.get_last_lr()[0]:.6f}',
                'episodes': f'{num_episodes:.0f}'
            })
        
        # 减少内存清理频率
        if (epoch + 1) % 100 == 0:
            torch.cuda.empty_cache()
    
    # 只在主进程保存模型
    if rank == 0:
        torch.save({
            'model_state_dict': model.module.state_dict(),
            'config': vars(config)
        }, 'model.pth')
        
        print("\n训练完成！模型已保存为 model.pth")
    
    # 清理进程组
    dist.destroy_process_group()