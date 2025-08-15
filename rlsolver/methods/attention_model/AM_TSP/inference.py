#!/usr/bin/env python
"""TSP Model Inference Script for evaluation."""

import os
import json
import time
import argparse
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

from models import TSPActor
from dataset import TSPDataset
from env import VectorizedTSPEnv
from utils import get_heuristic_solution
import config as default_config


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        model: Loaded TSPActor model
        config: Model configuration
    """
    # Try to load config from checkpoint directory
    config_path = Path(checkpoint_path).parent / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
            model_config = saved_config['model_config']
    else:
        # Fall back to default config
        print("Warning: No config.json found, using default configuration")
        model_config = {
            'embedding_size': default_config.EMBEDDING_SIZE,
            'hidden_size': default_config.HIDDEN_SIZE,
            'seq_len': default_config.SEQ_LEN,
            'n_head': default_config.N_HEAD,
            'C': default_config.C,
        }
    
    # Create model
    model = TSPActor(
        embedding_size=model_config['embedding_size'],
        hidden_size=model_config['hidden_size'],
        seq_len=model_config['seq_len'],
        n_head=model_config['n_head'],
        C=model_config['C']
    ).to(device)
    
    # Load checkpoint
    if checkpoint_path.endswith('.pth'):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        raise ValueError(f"Invalid checkpoint path: {checkpoint_path}")
    
    model.eval()
    return model, model_config


def rollout_batch(model, nodes, device='cuda', greedy=False):
    """Perform batch rollout for TSP instances.
    
    Args:
        model: TSPActor model
        nodes: [batch_size, seq_len, 2] node coordinates
        device: Device to run on
        greedy: If True, use greedy decoding
        
    Returns:
        tour_lengths: [batch_size] tour lengths
        tours: [batch_size, seq_len] tour indices
    """
    batch_size = nodes.size(0)
    seq_len = nodes.size(1)
    
    # Pre-compute encoder embeddings once
    with torch.no_grad():
        embedded = model.network.embedding(nodes)
        encoded = model.network.encoder(embedded)
    
    # Initialize state
    visited_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    current_node = None
    first_node = None
    tours = []
    
    # Rollout loop
    for step in range(seq_len):
        obs = {
            'nodes': nodes,
            'current_node': current_node,
            'first_node': first_node,
            'action_mask': ~visited_mask,
            'encoded': encoded  # Use cached encoding
        }
        
        with torch.no_grad():
            if greedy:
                # Greedy selection
                logits = model.network(obs)
                action = logits.argmax(dim=-1)
            else:
                # Sample from policy
                action, _ = model.get_action(obs, deterministic=False)
        
        # Update state
        if current_node is None:
            first_node = action.clone()
        current_node = action
        
        # Update visited mask
        batch_indices = torch.arange(batch_size, device=device)
        visited_mask[batch_indices, action] = True
        tours.append(action)
    
    # Stack tour indices
    tours = torch.stack(tours, dim=1)
    
    # Compute tour lengths
    vec_env = VectorizedTSPEnv(nodes, device=device)
    tour_lengths = vec_env.compute_all_tours(tours)
    
    return tour_lengths, tours


def evaluate_model(model, test_dataset, batch_size=256, device='cuda', 
                  use_heuristic=True, num_samples=10, greedy=False):
    """Evaluate model on test dataset.
    
    Args:
        model: TSPActor model
        test_dataset: TSPDataset to evaluate on
        batch_size: Batch size for evaluation
        device: Device to run on
        use_heuristic: If True, compare with heuristic solution
        num_samples: Number of samples per instance (for stochastic policy)
        greedy: If True, use greedy decoding
        
    Returns:
        results: Dictionary with evaluation metrics
    """
    model.eval()
    
    # Prepare data
    all_nodes = torch.stack([test_dataset[i][1] for i in range(len(test_dataset))])
    num_instances = len(test_dataset)
    
    # Results storage
    model_lengths = []
    heuristic_lengths = []
    
    # Compute heuristic solutions if requested
    if use_heuristic:
        print("Computing heuristic solutions...")
        for i in tqdm(range(num_instances)):
            nodes = all_nodes[i]
            heur_len = get_heuristic_solution(nodes)
            if heur_len is not None:
                heuristic_lengths.append(heur_len)
            else:
                print("Warning: elkai not available, skipping heuristic comparison")
                use_heuristic = False
                break
        
        if use_heuristic:
            heuristic_lengths = torch.tensor(heuristic_lengths)
    
    # Evaluate model
    print(f"Evaluating model ({'greedy' if greedy else f'sampling {num_samples} times'})...")
    
    for batch_start in tqdm(range(0, num_instances, batch_size)):
        batch_end = min(batch_start + batch_size, num_instances)
        batch_nodes = all_nodes[batch_start:batch_end].to(device)
        batch_size_actual = batch_nodes.size(0)
        
        if greedy or num_samples == 1:
            # Single rollout per instance
            lengths, _ = rollout_batch(model, batch_nodes, device, greedy=greedy)
            model_lengths.append(lengths.cpu())
        else:
            # Multiple samples per instance
            batch_best_lengths = torch.full((batch_size_actual,), float('inf'))
            
            for _ in range(num_samples):
                lengths, _ = rollout_batch(model, batch_nodes, device, greedy=False)
                batch_best_lengths = torch.minimum(batch_best_lengths, lengths.cpu())
            
            model_lengths.append(batch_best_lengths)
    
    # Concatenate results
    model_lengths = torch.cat(model_lengths)
    
    # Compute statistics
    results = {
        'num_instances': num_instances,
        'model_mean_length': model_lengths.mean().item(),
        'model_std_length': model_lengths.std().item(),
        'model_min_length': model_lengths.min().item(),
        'model_max_length': model_lengths.max().item(),
    }
    
    if use_heuristic:
        gaps = model_lengths / heuristic_lengths
        results.update({
            'heuristic_mean_length': heuristic_lengths.mean().item(),
            'mean_gap': gaps.mean().item(),
            'std_gap': gaps.std().item(),
            'min_gap': gaps.min().item(),
            'max_gap': gaps.max().item(),
            'num_better_than_heuristic': (gaps < 1.0).sum().item(),
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='TSP Model Inference')
    parser.add_argument('--checkpoint', type=str, default='model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--num_nodes', type=int, default=30,
                       help='Number of nodes in TSP instances')
    parser.add_argument('--num_test', type=int, default=2000,
                       help='Number of test instances')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=1,
                       help='Number of samples per instance (1 for greedy)')
    parser.add_argument('--greedy', action='store_true',
                       help='Use greedy decoding instead of sampling')
    parser.add_argument('--no_heuristic', action='store_true',
                       help='Skip heuristic comparison')
    parser.add_argument('--seed', type=int, default=111,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    device = torch.device(args.device)
    
    # Set random seed
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, device)
    
    # Verify node count matches model
    if config['seq_len'] != args.num_nodes:
        print(f"Warning: Model trained on {config['seq_len']} nodes, "
              f"but testing on {args.num_nodes} nodes")
        args.num_nodes = config['seq_len']
    
    # Create test dataset
    print(f"Creating test dataset with {args.num_test} instances...")
    test_dataset = TSPDataset(args.num_nodes, args.num_test, random_seed=args.seed)
    
    # Evaluate model
    print("\nStarting evaluation...")
    start_time = time.time()
    
    results = evaluate_model(
        model, 
        test_dataset,
        batch_size=args.batch_size,
        device=device,
        use_heuristic=not args.no_heuristic,
        num_samples=args.num_samples,
        greedy=args.greedy
    )
    
    eval_time = time.time() - start_time
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Number of instances: {results['num_instances']}")
    print(f"Model mean tour length: {results['model_mean_length']:.4f}")
    print(f"Model std tour length: {results['model_std_length']:.4f}")
    print(f"Model min tour length: {results['model_min_length']:.4f}")
    print(f"Model max tour length: {results['model_max_length']:.4f}")
    
    if 'mean_gap' in results:
        print("\nComparison with heuristic (LKH):")
        print(f"Heuristic mean length: {results['heuristic_mean_length']:.4f}")
        print(f"Mean gap (model/heuristic): {results['mean_gap']:.4f}x")
        print(f"Std gap: {results['std_gap']:.4f}")
        print(f"Min gap: {results['min_gap']:.4f}x")
        print(f"Max gap: {results['max_gap']:.4f}x")
        print(f"Instances better than heuristic: {results['num_better_than_heuristic']}/{results['num_instances']}")
    
    print(f"\nEvaluation time: {eval_time:.2f} seconds")
    print(f"Time per instance: {eval_time/results['num_instances']*1000:.2f} ms")
    
    # Save results
    results_path = Path(args.checkpoint).parent / 'inference_results.json'
    results['eval_time'] = eval_time
    results['args'] = vars(args)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()