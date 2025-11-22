"""
Inference script for Graph Coloring problem using PIGNN (Physics-Inspired Graph Neural Network).

This script implements the inference pipeline for the Graph Coloring optimization problem
using pre-trained PIGNN models with PyTorch Lightning and PyTorch Geometric frameworks.
"""

import os
import random
import torch
import numpy as np
from tqdm import tqdm
from time import time
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data
from typing import Optional, List, Tuple

# Use new unified environment structure
from rlsolver.methods.PIGNN.data import DRegDataset
from rlsolver.methods.PIGNN.model import PIGNN
from rlsolver.methods.PIGNN.util import eval_graph_coloring
from rlsolver.methods.PIGNN.config import *
from rlsolver.methods.config import Problem


def temperature_sampling_decode(pred, edge_index, num_colors, temperature=TEMPERATURE, trials=TRIALS):
    """Temperature sampling decoding for a single graph."""
    device = pred.device
    num_nodes = pred.size(0)

    best_colors = None
    best_violations = float('inf')
    best_used_colors = float('inf')

    # Generate probability distribution for colors
    # Use sigmoid activation and scale to [0, num_colors-1]
    if pred.dim() == 1:
        color_probs = torch.sigmoid(pred).unsqueeze(-1).repeat(1, num_colors)
    else:
        color_probs = torch.sigmoid(pred)

    for trial in range(trials):
        # Temperature sampling
        if temperature > 0:
            # Apply temperature and sample
            scaled_probs = color_probs / temperature
            uniform_probs = torch.rand(num_nodes, num_colors, device=device)
            colors = (scaled_probs > uniform_probs).sum(dim=-1)
        else:
            # Greedy assignment
            colors = torch.argmax(color_probs, dim=-1)

        # Clip colors to valid range
        colors = torch.clamp(colors, 0, num_colors - 1)

        # Count constraint violations
        violations = 0
        for edge in edge_index.t():
            if colors[edge[0]] == colors[edge[1]]:
                violations += 1

        used_colors = len(torch.unique(colors))

        # Update best solution
        if (violations < best_violations) or \
           (violations == best_violations and used_colors < best_used_colors):
            best_violations = violations
            best_used_colors = used_colors
            best_colors = colors.clone()

    return best_colors, best_violations, best_used_colors


def _to_batch(data) -> Batch:
    if isinstance(data, Batch):
        return data
    if isinstance(data, Data):
        return Batch.from_data_list([data])
    raise TypeError(f"Unsupported batch type: {type(data)}")


def _split_batch_predictions(batch, predictions: torch.Tensor) -> List[Tuple[Data, torch.Tensor]]:
    batch = _to_batch(batch)
    graphs = batch.to_data_list()
    pointer = 0
    per_graph = []

    for data in graphs:
        data = data.to(predictions.device)
        num_nodes = data.num_nodes
        graph_pred = predictions[pointer:pointer + num_nodes]
        pointer += num_nodes
        per_graph.append((data, graph_pred))

    return per_graph


def load_model(model_path: str, device: torch.device) -> PIGNN:
    """
    Load a pre-trained PIGNN model for Graph Coloring inference.

    Args:
        model_path: Path to the saved model weights
        device: Device to load the model on

    Returns:
        Loaded PIGNN model
    """
    # Create model with same architecture as training
    model = PIGNN(
        in_dim=IN_DIM,  # Use specific feature dimension for graph coloring
        hidden_dim=round(IN_DIM / 2),
        problem=Problem.graph_coloring,
        lr=TRAIN_LEARNING_RATE,  # Not used during inference
        out_dim=NUM_COLORS,  # Multi-color output
        num_heads=NUM_HEADS,
        layer_type=GNN_MODEL
    )

    # Load pre-trained weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded successfully from: {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model.to(device)
    model.eval()
    return model


def create_inference_dataloader(num_nodes: int, num_graphs: int = 100) -> DataLoader:
    """
    Create a dataloader for inference.

    Args:
        num_nodes: Number of nodes in the graphs
        num_graphs: Number of graphs to generate for inference

    Returns:
        DataLoader for inference
    """
    # Create dataset for inference using the same feature dimension as training
    dataset = DRegDataset(NODE_DEGREE, num_graphs, num_nodes, IN_DIM, INFERENCE_SEED)

    dataloader = DataLoader(
        dataset.data,
        batch_size=INFERENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=INFERENCE_NUM_WORKERS
    )

    return dataloader


def run_inference(model: PIGNN, dataloader: DataLoader, device: torch.device) -> tuple:
    """
    Run inference on the given dataloader using the loaded model.

    Args:
        model: Loaded PIGNN model
        dataloader: DataLoader containing graphs for inference
        device: Device to run inference on

    Returns:
        Tuple of (energies, approximation_ratios, predictions)
    """
    energies, approximation_ratios, all_predictions = [], [], []

    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Processing batches'):
            # Move batch to device
            batch = batch.to(device)
            x, edge_index = batch.x, batch.edge_index

            # Get model predictions
            pred = model(x, edge_index)

            for graph_data, graph_pred in _split_batch_predictions(batch, pred):
                colors, violations, used_colors = temperature_sampling_decode(
                    graph_pred, graph_data.edge_index, NUM_COLORS, TEMPERATURE, TRIALS
                )

                energy, approx_ratio = eval_graph_coloring(
                    graph_data.edge_index, colors, NUM_COLORS, graph_data.num_nodes
                )
                energies.append(energy.item())
                approximation_ratios.append(approx_ratio.item())
                all_predictions.append(colors.detach().cpu())

    return energies, approximation_ratios, all_predictions


def save_results(energies: List[float], approximation_ratios: List[float],
                 predictions: List, output_dir: str = RESULTS_DIR):
    """
    Save inference results to files.

    Args:
        energies: List of energy values
        approximation_ratios: List of approximation ratios
        predictions: List of model predictions (color assignments)
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save numerical results
    results_file = os.path.join(output_dir, "graph_coloring_inference_results.txt")
    with open(results_file, 'w') as f:
        f.write("Graph Coloring Inference Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Number of graphs: {len(energies)}\n")
        f.write(f"Number of colors: {NUM_COLORS}\n")
        f.write(f"Temperature: {TEMPERATURE}\n")
        f.write(f"Sampling trials: {TRIALS}\n")
        f.write(f"Average energy: {np.mean(energies):.4f} ± {np.std(energies):.4f}\n")
        f.write(f"Average approximation ratio: {np.mean(approximation_ratios):.4f} ± {np.std(approximation_ratios):.4f}\n")
        f.write(f"Best energy: {np.min(energies):.4f}\n")
        f.write(f"Best approximation ratio: {np.max(approximation_ratios):.4f}\n")

    # Save detailed results
    detailed_file = os.path.join(output_dir, "graph_coloring_detailed_results.txt")
    with open(detailed_file, 'w') as f:
        for i, (energy, approx_ratio) in enumerate(zip(energies, approximation_ratios)):
            f.write(f"Graph {i+1}: Energy={energy:.4f}, ApproxRatio={approx_ratio:.4f}\n")

    # Save color assignments if needed (optional for large datasets)
    if len(predictions) <= 1000:  # Only save if not too large
        predictions_file = os.path.join(output_dir, "graph_coloring_predictions.pt")
        torch.save(predictions, predictions_file)
        print(f"Color assignments saved to: {predictions_file}")

    print(f"Results saved to: {output_dir}")


def run():
    """
    Main inference function for Graph Coloring problem.

    This function implements the complete inference pipeline:
    1. Model loading and setup
    2. Dataset creation for inference
    3. Inference execution with temperature sampling
    4. Results analysis and saving
    """
    # Set problem type explicitly to Graph Coloring
    globals()['PROBLEM'] = Problem.graph_coloring

    # Seed setup for reproducibility
    os.environ["PL_GLOBAL_SEED"] = str(INFERENCE_SEED)
    random.seed(INFERENCE_SEED)
    np.random.seed(INFERENCE_SEED)
    torch.manual_seed(INFERENCE_SEED)
    torch.cuda.manual_seed_all(INFERENCE_SEED)

    print("=" * 60)
    print("PIGNN Graph Coloring Inference")
    print("=" * 60)
    print(f"Graph Coloring configuration:")
    print(f"  Number of colors: {NUM_COLORS}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Sampling trials: {TRIALS}")
    print(f"  Feature dimension: {IN_DIM}")

    # Configure device
    device = torch.device(f'cuda:{INFERENCE_GPU_NUM}' if INFERENCE_GPU_NUM >= 0 and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    from rlsolver.methods.PIGNN.config import get_model_load_path
    model_path = get_model_load_path(problem="graph_coloring", num_nodes=INFERENCE_NUM_NODES)
    model = load_model(model_path, device)

    # Determine inference parameters
    if isinstance(INFERENCE_NUM_NODES, list):
        print(f"Running inference on multiple node sizes: {INFERENCE_NUM_NODES}")
        all_energies, all_ratios = [], []

        for num_nodes in INFERENCE_NUM_NODES:
            print(f"\nProcessing {num_nodes} nodes...")
            dataloader = create_inference_dataloader(num_nodes)
            energies, ratios, _ = run_inference(model, dataloader, device)

            all_energies.extend(energies)
            all_ratios.extend(ratios)

            print(f"Results for {num_nodes} nodes:")
            print(f"  Average energy: {np.mean(energies):.4f}")
            print(f"  Average approximation ratio: {np.mean(ratios):.4f}")

        energies = all_energies
        approximation_ratios = all_ratios
    else:
        print(f"Running inference on {INFERENCE_NUM_NODES} nodes...")
        dataloader = create_inference_dataloader(INFERENCE_NUM_NODES)
        energies, approximation_ratios, predictions = run_inference(model, dataloader, device)

    # Final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Total graphs processed: {len(energies)}")
    print(f"Number of colors used: {NUM_COLORS}")
    print(f"Temperature sampling: {TEMPERATURE}")
    print(f"Average energy: {np.mean(energies):.4f} ± {np.std(energies):.4f}")
    print(f"Average approximation ratio: {np.mean(approximation_ratios):.4f} ± {np.std(approximation_ratios):.4f}")
    print(f"Best energy: {np.min(energies):.4f}")
    print(f"Best approximation ratio: {np.max(approximation_ratios):.4f}")

    # Save results
    if 'predictions' in locals():
        save_results(energies, approximation_ratios, predictions)
    else:
        save_results(energies, approximation_ratios, [])

    print("Inference completed successfully!")


if __name__ == '__main__':
    run()
