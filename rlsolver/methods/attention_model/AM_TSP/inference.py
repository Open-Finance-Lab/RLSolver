# inference.py

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os
import json

from models import TSPActor
from dataset import TSPDataset
from env import VectorizedTSPEnv
from util import get_heuristic_solution
from config import *

def load_model(model_path, device):
    """Load trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        device: Device string or torch.device
    
    Returns:
        model: Loaded TSPActor model
    """
    model = TSPActor(
        embedding_size=EMBEDDING_SIZE,
        hidden_size=HIDDEN_SIZE,
        seq_len=NUM_NODES,
        n_head=N_HEAD,
        C=C
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


@torch.no_grad()
def rollout_inference(model, nodes, num_rollouts=None, device=None):
    """POMO rollout for inference - optimized version.
    
    Args:
        model: TSPActor model
        nodes: [batch_size, seq_len, 2]
        num_rollouts: Number of parallel rollouts (default: seq_len for POMO)
        device: Device string or torch.device
    
    Returns:
        best_tours: [batch_size, seq_len] - best tour for each instance
        best_lengths: [batch_size] - length of best tours
    """
    if device is None:
        device = nodes.device
        
    batch_size = nodes.size(0)
    seq_len = nodes.size(1)
    if num_rollouts is None:
        num_rollouts = seq_len
    
    # Pre-compute encoder embeddings once
    embedded = model.network.embedding(nodes)
    encoded = model.network.encoder(embedded)
    
    # Expand for parallel rollouts
    encoded_expanded = encoded.unsqueeze(1).expand(batch_size, num_rollouts, seq_len, -1)
    nodes_expanded = nodes.unsqueeze(1).expand(batch_size, num_rollouts, seq_len, 2)
    
    encoded_flat = encoded_expanded.contiguous().view(batch_size * num_rollouts, seq_len, -1)
    nodes_flat = nodes_expanded.contiguous().view(batch_size * num_rollouts, seq_len, 2)
    
    # Initialize states
    visited_mask = torch.zeros(batch_size * num_rollouts, seq_len, dtype=torch.bool, device=device)
    flat_indices = torch.arange(batch_size * num_rollouts, device=device)
    
    # POMO: Different starting nodes
    pomo_indices = torch.arange(num_rollouts, device=device).repeat(batch_size) % seq_len
    first_node = pomo_indices
    current_node = pomo_indices
    visited_mask[flat_indices, current_node] = True
    
    # Store actions
    actions_tensor = torch.empty(batch_size * num_rollouts, seq_len, dtype=torch.long, device=device)
    actions_tensor[:, 0] = current_node
    
    # Rollout
    for step in range(1, seq_len):
        obs = {
            'nodes': nodes_flat,
            'current_node': current_node,
            'first_node': first_node,
            'action_mask': ~visited_mask,
            'encoded': encoded_flat
        }
        
        # Greedy action selection for inference
        logits = model.network(obs)
        action = logits.argmax(dim=-1)
        
        current_node = action
        visited_mask[flat_indices, action] = True
        actions_tensor[:, step] = action
    
    # Compute tour lengths
    vec_env = VectorizedTSPEnv(nodes_flat, device=device)
    tour_lengths = vec_env.compute_all_tours(actions_tensor)
    
    # Reshape and find best tours
    tour_lengths = tour_lengths.view(batch_size, num_rollouts)
    actions = actions_tensor.view(batch_size, num_rollouts, seq_len)
    
    # Get best tour for each instance
    best_indices = tour_lengths.argmin(dim=1)
    best_lengths = tour_lengths.gather(1, best_indices.unsqueeze(1)).squeeze(1)
    best_tours = actions.gather(1, best_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, seq_len)).squeeze(1)
    
    return best_tours, best_lengths


def inference_on_dataset(model, dataset, batch_size=None, device=None):
    """Run inference on entire dataset.
    
    Args:
        model: TSPActor model
        dataset: TSPDataset
        batch_size: Batch size for inference
        device: Device string or torch.device
    
    Returns:
        all_tours: List of best tours
        all_lengths: Tensor of tour lengths
        heuristic_gap: Average gap vs heuristic (if available)
    """
    if batch_size is None:
        batch_size = INFERENCE_BATCH_SIZE
    
    if device is None:
        device = INFERENCE_DEVICE
    
    all_tours = []
    all_lengths = []
    
    # Create batches
    num_samples = len(dataset)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"Running inference on {num_samples} instances...")
    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        
        # Gather batch
        batch_nodes = []
        for i in range(start_idx, end_idx):
            _, nodes = dataset[i]
            batch_nodes.append(nodes)
        
        batch_nodes = torch.stack(batch_nodes).to(device)
        
        # Run inference
        tours, lengths = rollout_inference(
            model, batch_nodes, 
            num_rollouts=NUM_INFERENCE_ENVS,
            device=device
        )
        
        all_tours.extend(tours.cpu().numpy())
        all_lengths.append(lengths.cpu())
    
    all_lengths = torch.cat(all_lengths)
    
    # Calculate heuristic gap if available
    heuristic_gap = None
    if COMPUTE_HEURISTIC_GAP:
        print("Computing heuristic solutions for comparison...")
        heuristic_lengths = []
        
        for i in tqdm(range(num_samples)):
            _, nodes = dataset[i]
            h_length = get_heuristic_solution(nodes)
            if h_length is not None:
                heuristic_lengths.append(h_length)
            else:
                print("Warning: elkai not available, skipping heuristic comparison")
                break
        
        if len(heuristic_lengths) == num_samples:
            heuristic_lengths = torch.tensor(heuristic_lengths)
            gap = (all_lengths / heuristic_lengths).mean().item()
            heuristic_gap = gap
            print(f"Average gap vs heuristic: {gap:.4f}x")
    
    return all_tours, all_lengths, heuristic_gap


def main():
    """Main inference function."""
    # Get device from config
    device = INFERENCE_DEVICE
    
    # Print configuration
    print("="*50)
    print("TSP Solver - Inference Mode")
    print("="*50)
    
    # Load model
    print(f"\nLoading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH, device)
    
    # Create test dataset
    test_dataset = TSPDataset(NUM_NODES, NUM_TEST_SAMPLES, TEST_SEED)
    
    # Run inference
    print(f"\nInference Configuration:")
    print(f"  Device: {device} (GPU ID: {INFERENCE_GPU_ID})")
    print(f"  Problem size: {NUM_NODES}")
    print(f"  Test samples: {NUM_TEST_SAMPLES}")
    print(f"  Inference batch size: {INFERENCE_BATCH_SIZE}")
    print(f"  Number of parallel rollouts (POMO): {NUM_INFERENCE_ENVS}")
    print(f"  Compute heuristic gap: {COMPUTE_HEURISTIC_GAP}")
    print(f"  Save results: {SAVE_RESULTS}")
    print()
    
    tours, lengths, gap = inference_on_dataset(
        model, test_dataset, 
        batch_size=INFERENCE_BATCH_SIZE,
        device=device
    )
    
    # Print statistics
    print(f"\nResults:")
    print(f"  Average tour length: {lengths.mean().item():.4f}")
    print(f"  Best tour length: {lengths.min().item():.4f}")
    print(f"  Worst tour length: {lengths.max().item():.4f}")
    print(f"  Std deviation: {lengths.std().item():.4f}")
    
    if gap is not None:
        print(f"  Gap vs heuristic: {gap:.4f}x")
    
    # Save results if requested
    if SAVE_RESULTS:
        results = {
            'tours': [tour.tolist() for tour in tours],
            'lengths': lengths.tolist(),
            'statistics': {
                'mean': lengths.mean().item(),
                'min': lengths.min().item(),
                'max': lengths.max().item(),
                'std': lengths.std().item(),
                'gap': gap
            },
            'config': {
                'seq_len': NUM_NODES,
                'num_samples': NUM_TEST_SAMPLES,
                'num_rollouts': NUM_INFERENCE_ENVS,
                'model_path': MODEL_PATH,
                'inference_gpu_id': INFERENCE_GPU_ID,
                'device': str(device)
            }
        }
        
        os.makedirs(RESULTS_DIR, exist_ok=True)
        results_path = os.path.join(RESULTS_DIR, RESULTS_FILENAME)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {results_path}")
        
    print("="*50)
    print("Inference completed successfully!")
    print("="*50)


if __name__ == '__main__':
    # Check if we're in inference mode
    if hasattr(args, 'TRAIN_MODE') and TRAIN_INFERENCE == 0:
        print("Warning: TRAIN_MODE is set to 0 (training). Set TRAIN_MODE=1 for inference.")
        print("Proceeding with inference anyway...")
    
    main()
