# inference.py

import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Import model components from your existing files
from models import TSPActor
from utils import get_heuristic_solution

# ================== Configuration ==================
# Hyperparameters - Adjust these as needed
NUM_NODES = 20  # Number of nodes in TSP problem (adjustable)
NUM_TEST_SAMPLES = 1000  # Number of test instances
TEST_SEED = 1234  # Random seed for test data generation
BATCH_SIZE = 64  # Batch size for inference
POMO_SIZE = NUM_NODES  # Number of parallel rollouts (typically = NUM_NODES for POMO)

# Model configuration (must match training)
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 128
N_HEAD = 8
C = 15.0 * (NUM_NODES / 20) ** 0.5

# Paths
MODEL_PATH = "checkpoints/best_model_epoch361.pth"  # Adjust filename as needed
RESULTS_DIR = "results"
COMPUTE_HEURISTIC = True  # Whether to compute heuristic solution for comparison

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_test_data(num_samples, num_nodes, seed=1234):
    """Generate test TSP instances with uniform random coordinates in [0, 1)."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    test_data = []
    for _ in range(num_samples):
        # Generate random 2D coordinates in [0, 1) x [0, 1)
        nodes = torch.rand(num_nodes, 2, generator=generator)
        test_data.append(nodes)
    
    return test_data


def compute_tour_lengths(nodes, actions):
    """Compute tour lengths for given nodes and actions.
    
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
    
    # Calculate distances between consecutive nodes
    diffs = tour_nodes[:, :, 1:] - tour_nodes[:, :, :-1]
    distances = torch.norm(diffs, dim=3)
    
    # Add distance from last to first
    last_to_first = torch.norm(tour_nodes[:, :, -1] - tour_nodes[:, :, 0], dim=2)
    
    lengths = distances.sum(dim=2) + last_to_first
    
    return lengths


def rollout_episode_pomo(model, nodes, pomo_size=None):
    """POMO rollout for inference (no gradients).
    
    Args:
        model: TSPActor model
        nodes: [batch_size, seq_len, 2]
        pomo_size: Number of parallel rollouts (default: seq_len)
    
    Returns:
        tour_lengths: [batch_size, pomo_size]
        actions: [batch_size, pomo_size, seq_len]
    """
    batch_size = nodes.size(0)
    seq_len = nodes.size(1)
    device = nodes.device
    
    if pomo_size is None:
        pomo_size = seq_len
    
    with torch.no_grad():
        # Pre-compute encoder embeddings ONCE (shared across all POMO)
        embedded = model.network.embedding(nodes)
        encoded = model.network.encoder(embedded)  # [batch, seq_len, embed_dim]
        
        # Initialize states - structured format
        visited_mask = torch.zeros(batch_size, pomo_size, seq_len, dtype=torch.bool, device=device)
        
        # POMO: Different starting nodes for each rollout
        pomo_indices = torch.arange(pomo_size, device=device) % seq_len
        first_node = pomo_indices.unsqueeze(0).expand(batch_size, -1)  # [batch, pomo]
        current_node = first_node.clone()
        
        # Mark first nodes as visited
        batch_idx = torch.arange(batch_size, device=device)[:, None].expand(-1, pomo_size)
        pomo_idx = torch.arange(pomo_size, device=device)[None, :].expand(batch_size, -1)
        visited_mask[batch_idx, pomo_idx, current_node] = True
        
        # Collect actions
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
            
            # Get greedy action (deterministic=True for inference)
            action, _ = model.get_action(obs, deterministic=True)
            
            current_node = action
            visited_mask[batch_idx, pomo_idx, action] = True
            
            actions_list.append(action)
        
        # Stack actions
        actions = torch.stack(actions_list, dim=2)  # [batch, pomo, seq_len]
        
        # Compute tour lengths
        tour_lengths = compute_tour_lengths(nodes, actions)
        
    return tour_lengths, actions


def load_model(model_path, num_nodes):
    """Load trained model from checkpoint."""
    # Create model
    model = TSPActor(
        embedding_size=EMBEDDING_SIZE,
        hidden_size=HIDDEN_SIZE,
        seq_len=num_nodes,
        n_head=N_HEAD,
        C=C
    ).to(DEVICE)
    
    # Load checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}")
    
    # Load state dict
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DDP training)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded successfully. Total parameters: {total_params:,}")
    
    return model


def run_inference():
    """Main inference function."""
    print("="*60)
    print("TSP Model Inference with POMO")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Number of nodes: {NUM_NODES}")
    print(f"  Number of test samples: {NUM_TEST_SAMPLES}")
    print(f"  Test seed: {TEST_SEED}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  POMO size: {POMO_SIZE}")
    print(f"  Device: {DEVICE}")
    print(f"  Model path: {MODEL_PATH}")
    print()
    
    # Load model
    model = load_model(MODEL_PATH, NUM_NODES)
    
    # Generate test data
    print(f"Generating {NUM_TEST_SAMPLES} test instances...")
    test_data = generate_test_data(NUM_TEST_SAMPLES, NUM_NODES, TEST_SEED)
    
    # Run inference in batches
    all_tour_lengths = []
    all_best_lengths = []
    all_actions = []
    batch_times = []  # Store time for each sample
    
    print("\nRunning inference...")
    num_batches = (NUM_TEST_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        # Get batch
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, NUM_TEST_SAMPLES)
        batch_nodes = torch.stack(test_data[start_idx:end_idx]).to(DEVICE)
        batch_size_actual = batch_nodes.size(0)
        
        # Time this batch
        batch_start_time = time.time()
        
        # Run POMO rollout
        tour_lengths, actions = rollout_episode_pomo(model, batch_nodes, POMO_SIZE)
        
        # Get best tour for each instance (minimum across POMO rollouts)
        best_lengths, best_indices = tour_lengths.min(dim=1)
        
        batch_time = time.time() - batch_start_time
        
        # Store time per sample for this batch
        time_per_sample = batch_time / batch_size_actual
        batch_times.extend([time_per_sample] * batch_size_actual)
        
        # Store results
        all_tour_lengths.append(tour_lengths.cpu())
        all_best_lengths.append(best_lengths.cpu())
        all_actions.append(actions.cpu())
    
    # Calculate inference time using only last 100 samples (to avoid compile overhead)
    num_samples_for_timing = min(100, NUM_TEST_SAMPLES)
    avg_time_per_sample = np.mean(batch_times[-num_samples_for_timing:])
    total_inference_time = sum(batch_times)
    
    # Concatenate results
    all_tour_lengths = torch.cat(all_tour_lengths, dim=0)
    all_best_lengths = torch.cat(all_best_lengths, dim=0)
    all_actions = torch.cat(all_actions, dim=0)
    
    # Calculate statistics
    mean_length = all_best_lengths.mean().item()
    std_length = all_best_lengths.std().item()
    min_length = all_best_lengths.min().item()
    max_length = all_best_lengths.max().item()
    
    print(f"\nInference completed in {total_inference_time:.2f} seconds total")
    print(f"Average time per instance (last {num_samples_for_timing} samples): {avg_time_per_sample*1000:.2f} ms")
    print(f"  Note: Timing based on last {num_samples_for_timing} samples to exclude compilation overhead")
    
    print(f"\nResults (best of {POMO_SIZE} POMO rollouts):")
    print(f"  Mean tour length: {mean_length:.4f}")
    print(f"  Std tour length: {std_length:.4f}")
    print(f"  Min tour length: {min_length:.4f}")
    print(f"  Max tour length: {max_length:.4f}")
    
    # Compare with heuristic if available
    if COMPUTE_HEURISTIC:
        try:
            print("\nComputing heuristic solutions for comparison...")
            heuristic_lengths = []
            
            for i in tqdm(range(min(100, NUM_TEST_SAMPLES)), desc="Computing heuristic"):
                heuristic_length = get_heuristic_solution(test_data[i])
                if heuristic_length is not None:
                    heuristic_lengths.append(heuristic_length)
            
            if heuristic_lengths:
                heuristic_lengths = torch.tensor(heuristic_lengths)
                model_lengths = all_best_lengths[:len(heuristic_lengths)]
                
                gap = (model_lengths / heuristic_lengths).mean().item()
                print(f"\nGap vs heuristic (on {len(heuristic_lengths)} samples): {gap:.4f}x")
                print(f"  (1.0 = optimal, >1.0 = suboptimal)")
        except Exception as e:
            print(f"\nCould not compute heuristic: {e}")
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {
        "config": {
            "num_nodes": NUM_NODES,
            "num_test_samples": NUM_TEST_SAMPLES,
            "test_seed": TEST_SEED,
            "batch_size": BATCH_SIZE,
            "pomo_size": POMO_SIZE,
            "model_path": MODEL_PATH,
        },
        "results": {
            "mean_tour_length": mean_length,
            "std_tour_length": std_length,
            "min_tour_length": min_length,
            "max_tour_length": max_length,
            "total_inference_time_seconds": total_inference_time,
            "avg_time_per_instance_ms": avg_time_per_sample*1000,
            "timing_note": f"Average based on last {num_samples_for_timing} samples (excluding compilation overhead)",
        },
        "timestamp": datetime.now().isoformat(),
    }
    
    results_path = os.path.join(RESULTS_DIR, f"inference_results_{NUM_NODES}nodes.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {results_path}")
    
    # Optionally save some sample tours for visualization
    sample_tours_path = os.path.join(RESULTS_DIR, f"sample_tours_{NUM_NODES}nodes.pth")
    torch.save({
        'nodes': test_data[:10],  # First 10 test instances
        'tours': all_actions[:10],  # Their tours
        'lengths': all_best_lengths[:10],  # Their lengths
    }, sample_tours_path)
    print(f"Sample tours saved to {sample_tours_path}")
    
    return results


def visualize_tour(nodes, tour, title="TSP Tour"):
    """Visualize a TSP tour (optional - requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
        
        if isinstance(nodes, torch.Tensor):
            nodes = nodes.cpu().numpy()
        if isinstance(tour, torch.Tensor):
            tour = tour.cpu().numpy()
        
        plt.figure(figsize=(8, 8))
        
        # Plot nodes
        plt.scatter(nodes[:, 0], nodes[:, 1], c='red', s=50, zorder=5)
        
        # Plot tour
        tour_nodes = nodes[tour]
        for i in range(len(tour_nodes)):
            start = tour_nodes[i]
            end = tour_nodes[(i + 1) % len(tour_nodes)]
            plt.plot([start[0], end[0]], [start[1], end[1]], 'b-', alpha=0.6)
        
        # Mark start node
        plt.scatter(nodes[tour[0], 0], nodes[tour[0], 1], c='green', s=100, marker='s', zorder=6)
        
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")


if __name__ == "__main__":
    # Run inference
    results = run_inference()
    
    # Optional: Visualize a sample tour
    # Uncomment the following to visualize the first test instance
    """
    print("\nVisualizing a sample tour...")
    sample_data = torch.load(os.path.join(RESULTS_DIR, f"sample_tours_{NUM_NODES}nodes.pth"))
    nodes = sample_data['nodes'][0]
    tours = sample_data['tours'][0]  # [pomo_size, seq_len]
    lengths = compute_tour_lengths(nodes.unsqueeze(0), tours.unsqueeze(0))[0]
    best_idx = lengths.argmin()
    best_tour = tours[best_idx]
    visualize_tour(nodes, best_tour, f"TSP Tour (Length: {lengths[best_idx]:.4f})")
    """