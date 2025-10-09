"""TSP Inference using TSPEnv."""

import torch
from tqdm import tqdm
import os
import json

from models import TSPActor
from dataset import TSPDataset
from env import TSPEnv
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
    
    num_samples = len(dataset)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"Running inference on {num_samples} instances...")
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            
            batch_nodes = []
            for i in range(start_idx, end_idx):
                _, nodes = dataset[i]
                batch_nodes.append(nodes)
            batch_nodes = torch.stack(batch_nodes).to(device)
            
            env = TSPEnv(batch_nodes, device=device)
            tours, lengths = env.rollout_greedy(
                model,
                num_rollouts=NUM_INFERENCE_ENVS
            )
            
            all_tours.extend(tours.cpu().numpy())
            all_lengths.append(lengths.cpu())
    
    all_lengths = torch.cat(all_lengths)
    
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
    device = INFERENCE_DEVICE
    
    print("="*50)
    print("TSP Solver - Inference Mode")
    print("="*50)
    
    print(f"\nLoading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH, device)
    
    test_dataset = TSPDataset(NUM_NODES, NUM_TEST_SAMPLES, TEST_SEED)
    
    print(f"\nInference Configuration:")
    print(f" Device: {device} (GPU ID: {INFERENCE_GPU_ID})")
    print(f" Problem size: {NUM_NODES}")
    print(f" Test samples: {NUM_TEST_SAMPLES}")
    print(f" Inference batch size: {INFERENCE_BATCH_SIZE}")
    print(f" Number of parallel rollouts (POMO): {NUM_INFERENCE_ENVS}")
    print(f" Compute heuristic gap: {COMPUTE_HEURISTIC_GAP}")
    print(f" Save results: {SAVE_RESULTS}")
    print()
    
    tours, lengths, gap = inference_on_dataset(
        model, test_dataset,
        batch_size=INFERENCE_BATCH_SIZE,
        device=device
    )
    
    print(f"\nResults:")
    print(f" Average tour length: {lengths.mean().item():.4f}")
    print(f" Best tour length: {lengths.min().item():.4f}")
    print(f" Worst tour length: {lengths.max().item():.4f}")
    print(f" Std deviation: {lengths.std().item():.4f}")
    if gap is not None:
        print(f" Gap vs heuristic: {gap:.4f}x")
    
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
    if hasattr(args, 'TRAIN_MODE') and TRAIN_INFERENCE == 0:
        print("Warning: TRAIN_MODE is set to 0 (training). Set TRAIN_MODE=1 for inference.")
        print("Proceeding with inference anyway...")
    main()