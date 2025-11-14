# inference.py

import os
import json
import time
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime

from models import TSPActor
from trainer import rollout_episode_pomo_structured
from utils import get_heuristic_solution
import config as args


def generate_test_data(num_samples, num_nodes, seed=1234):
    """Generate test TSP instances."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    test_data = []
    for _ in range(num_samples):
        nodes = torch.rand(num_nodes, 2, generator=generator)
        test_data.append(nodes)
    return test_data


def load_model(model_path, num_nodes, device):
    """Load trained model from checkpoint."""
    model = TSPActor(
        embedding_size=args.EMBEDDING_SIZE,
        hidden_size=args.HIDDEN_SIZE,
        seq_len=num_nodes,
        n_head=args.N_HEAD,
        C=args.C
    ).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state_dict', checkpoint)
    else:
        state_dict = checkpoint

    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded successfully. Total parameters: {total_params:,}")

    return model


def run_inference(model_path=None, num_nodes=None):
    """Main inference function."""
    model_path = model_path or args.MODEL_PATH
    num_nodes = num_nodes or args.NUM_NODES
    device = args.INFERENCE_DEVICE

    print("="*60)
    print("TSP Model Inference with POMO")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Number of nodes: {num_nodes}")
    print(f"  Number of test samples: {args.NUM_TEST_SAMPLES}")
    print(f"  Batch size: {args.INFERENCE_BATCH_SIZE}")
    print(f"  POMO size: {args.NUM_INFERENCE_POMO}")
    print(f"  Device: {device}")
    print(f"  Model path: {model_path}")
    print()

    model = load_model(model_path, num_nodes, device)

    print(f"Generating {args.NUM_TEST_SAMPLES} test instances...")
    test_data = generate_test_data(args.NUM_TEST_SAMPLES, num_nodes, args.TEST_SEED)

    all_tour_lengths = []
    all_best_lengths = []
    all_actions = []
    batch_times = []

    print("\nRunning inference...")
    num_batches = (args.NUM_TEST_SAMPLES + args.INFERENCE_BATCH_SIZE - 1) // args.INFERENCE_BATCH_SIZE

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * args.INFERENCE_BATCH_SIZE
        end_idx = min(start_idx + args.INFERENCE_BATCH_SIZE, args.NUM_TEST_SAMPLES)
        batch_nodes = torch.stack(test_data[start_idx:end_idx]).to(device)
        batch_size_actual = batch_nodes.size(0)

        batch_start_time = time.time()
        tour_lengths, _, actions = rollout_episode_pomo_structured(
            model, batch_nodes, args.NUM_INFERENCE_POMO, device
        )
        best_lengths = tour_lengths.min(dim=1)[0]
        batch_time = time.time() - batch_start_time
        time_per_sample = batch_time / batch_size_actual
        batch_times.extend([time_per_sample] * batch_size_actual)

        all_tour_lengths.append(tour_lengths.cpu())
        all_best_lengths.append(best_lengths.cpu())
        all_actions.append(actions.cpu())

    num_samples_for_timing = min(100, args.NUM_TEST_SAMPLES)
    avg_time_per_sample = np.mean(batch_times[-num_samples_for_timing:])
    total_inference_time = sum(batch_times)

    all_tour_lengths = torch.cat(all_tour_lengths, dim=0)
    all_best_lengths = torch.cat(all_best_lengths, dim=0)
    all_actions = torch.cat(all_actions, dim=0)

    mean_length = all_best_lengths.mean().item()
    std_length = all_best_lengths.std().item()
    min_length = all_best_lengths.min().item()
    max_length = all_best_lengths.max().item()

    print(f"\nInference completed in {total_inference_time:.2f} seconds")
    print(f"Average time per instance (last {num_samples_for_timing} samples): {avg_time_per_sample*1000:.2f} ms")
    print(f"\nResults (best of {args.NUM_INFERENCE_POMO} POMO rollouts):")
    print(f"  Mean tour length: {mean_length:.4f}")
    print(f"  Std tour length: {std_length:.4f}")
    print(f"  Min tour length: {min_length:.4f}")
    print(f"  Max tour length: {max_length:.4f}")

    if args.COMPUTE_HEURISTIC_GAP:
        try:
            print("\nComputing heuristic solutions for comparison...")
            heuristic_lengths = []
            for i in tqdm(range(min(100, args.NUM_TEST_SAMPLES)), desc="Computing heuristic"):
                heuristic_length = get_heuristic_solution(test_data[i])
                if heuristic_length is not None:
                    heuristic_lengths.append(heuristic_length)
                else:
                    break

            if heuristic_lengths:
                heuristic_lengths = torch.tensor(heuristic_lengths)
                model_lengths = all_best_lengths[:len(heuristic_lengths)]
                gap = (model_lengths / heuristic_lengths).mean().item()
                print(f"\nGap vs heuristic: {gap:.4f}x (1.0 = optimal)")
        except Exception as e:
            print(f"\nCould not compute heuristic: {e}")

    if args.SAVE_RESULTS:
        os.makedirs(args.RESULTS_DIR, exist_ok=True)
        results = {
            "config": {
                "num_nodes": num_nodes,
                "num_test_samples": args.NUM_TEST_SAMPLES,
                "batch_size": args.INFERENCE_BATCH_SIZE,
                "pomo_size": args.NUM_INFERENCE_POMO,
                "model_path": model_path,
            },
            "results": {
                "mean_tour_length": mean_length,
                "std_tour_length": std_length,
                "min_tour_length": min_length,
                "max_tour_length": max_length,
                "total_inference_time_seconds": total_inference_time,
                "avg_time_per_instance_ms": avg_time_per_sample * 1000,
            },
            "timestamp": datetime.now().isoformat(),
        }

        results_path = os.path.join(args.RESULTS_DIR, f"inference_results_{num_nodes}nodes.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {results_path}")

        sample_tours_path = os.path.join(args.RESULTS_DIR, f"sample_tours_{num_nodes}nodes.pth")
        torch.save({
            'nodes': test_data[:10],
            'tours': all_actions[:10],
            'lengths': all_best_lengths[:10],
        }, sample_tours_path)
        print(f"Sample tours saved to {sample_tours_path}")

    return results


def visualize_tour(nodes, tour, title="TSP Tour"):
    """Visualize a TSP tour (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
        nodes = nodes.cpu().numpy() if isinstance(nodes, torch.Tensor) else nodes
        tour = tour.cpu().numpy() if isinstance(tour, torch.Tensor) else tour

        plt.figure(figsize=(8, 8))
        plt.scatter(nodes[:, 0], nodes[:, 1], c='red', s=50, zorder=5)

        tour_nodes = nodes[tour]
        for i in range(len(tour_nodes)):
            start = tour_nodes[i]
            end = tour_nodes[(i + 1) % len(tour_nodes)]
            plt.plot([start[0], end[0]], [start[1], end[1]], 'b-', alpha=0.6)

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
    results = run_inference()
