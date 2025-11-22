"""
Inference script for MaxCut problem using PIGNN (Physics-Inspired Graph Neural Network).

This script implements the inference pipeline for the MaxCut optimization problem
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
from rlsolver.methods.PIGNN.util import eval_maxcut
from rlsolver.methods.PIGNN.config import *
from rlsolver.methods.config import Problem


def load_model(model_path: str, device: torch.device) -> PIGNN:
    """
    Load a pre-trained PIGNN model for MaxCut inference.

    Args:
        model_path: Path to the saved model weights
        device: Device to load the model on

    Returns:
        Loaded PIGNN model
    """
    # Create model with same architecture as training
    in_dim = INFERENCE_NUM_NODES if hasattr(INFERENCE_NUM_NODES, '__len__') else INFERENCE_NUM_NODES
    if isinstance(in_dim, list):
        in_dim = max(in_dim)  # Use maximum for flexibility

    # Calculate feature dimension
    feature_dim = int(in_dim ** 0.5)
    feature_dim = round(feature_dim)
    hidden_dim = round(feature_dim / 2)

    model = PIGNN(
        in_dim=feature_dim,
        hidden_dim=hidden_dim,
        problem=Problem.maxcut,
        lr=TRAIN_LEARNING_RATE,  # Not used during inference
        out_dim=1,
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
    # Calculate feature dimension
    feature_dim = int(num_nodes ** 0.5)
    feature_dim = round(feature_dim)

    # Create dataset for inference
    dataset = DRegDataset(NODE_DEGREE, num_graphs, num_nodes, feature_dim, INFERENCE_SEED)

    dataloader = DataLoader(
        dataset.data,
        batch_size=INFERENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=INFERENCE_NUM_WORKERS
    )

    return dataloader


def _to_batch(data) -> Batch:
    if isinstance(data, Batch):
        return data
    if isinstance(data, Data):
        return Batch.from_data_list([data])
    raise TypeError(f"Unsupported batch type: {type(data)}")


def _split_batch_predictions(batch, predictions: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Split batched predictions into per-graph tensors."""
    batch = _to_batch(batch)
    graphs = batch.to_data_list()
    pointer = 0
    per_graph = []

    preds = predictions.squeeze(-1)
    for data in graphs:
        data = data.to(preds.device)
        num_nodes = data.num_nodes
        graph_pred = preds[pointer:pointer + num_nodes]
        pointer += num_nodes
        per_graph.append((data.edge_index, graph_pred))

    return per_graph


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
            proj = torch.round(pred)

            for graph_edge_index, graph_pred in _split_batch_predictions(batch, proj):
                energy, approx_ratio = eval_maxcut(
                    graph_edge_index, graph_pred, NODE_DEGREE, graph_pred.size(0)
                )
                energies.append(energy.item())
                approximation_ratios.append(approx_ratio.item())
                all_predictions.append(graph_pred.detach().cpu())

    return energies, approximation_ratios, all_predictions


def save_results(energies: List[float], approximation_ratios: List[float],
                 predictions: List, output_dir: str = RESULTS_DIR):
    """
    Save inference results to files.

    Args:
        energies: List of energy values
        approximation_ratios: List of approximation ratios
        predictions: List of model predictions
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save numerical results
    results_file = os.path.join(output_dir, "maxcut_inference_results.txt")
    with open(results_file, 'w') as f:
        f.write("MaxCut Inference Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Number of graphs: {len(energies)}\n")
        f.write(f"Average energy: {np.mean(energies):.4f} ± {np.std(energies):.4f}\n")
        f.write(f"Average approximation ratio: {np.mean(approximation_ratios):.4f} ± {np.std(approximation_ratios):.4f}\n")
        f.write(f"Best energy: {np.min(energies):.4f}\n")
        f.write(f"Best approximation ratio: {np.max(approximation_ratios):.4f}\n")

    # Save detailed results
    detailed_file = os.path.join(output_dir, "maxcut_detailed_results.txt")
    with open(detailed_file, 'w') as f:
        for i, (energy, approx_ratio) in enumerate(zip(energies, approximation_ratios)):
            f.write(f"Graph {i+1}: Energy={energy:.4f}, ApproxRatio={approx_ratio:.4f}\n")

    # Save predictions if needed (optional for large datasets)
    if len(predictions) <= 1000:  # Only save if not too large
        predictions_file = os.path.join(output_dir, "maxcut_predictions.pt")
        torch.save(predictions, predictions_file)
        print(f"Predictions saved to: {predictions_file}")

    print(f"Results saved to: {output_dir}")


def run():
    """
    Main inference function for MaxCut problem.

    This function implements the complete inference pipeline:
    1. Model loading and setup
    2. Dataset creation for inference
    3. Inference execution
    4. Results analysis and saving
    """
    # Set problem type explicitly to MaxCut
    globals()['PROBLEM'] = Problem.maxcut

    # Seed setup for reproducibility
    os.environ["PL_GLOBAL_SEED"] = str(INFERENCE_SEED)
    random.seed(INFERENCE_SEED)
    np.random.seed(INFERENCE_SEED)
    torch.manual_seed(INFERENCE_SEED)
    torch.cuda.manual_seed_all(INFERENCE_SEED)

    print("=" * 60)
    print("PIGNN MaxCut Inference")
    print("=" * 60)

    # Configure device
    device = torch.device(f'cuda:{INFERENCE_GPU_NUM}' if INFERENCE_GPU_NUM >= 0 and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    from rlsolver.methods.PIGNN.config import get_model_load_path
    model_path = get_model_load_path(problem="maxcut", num_nodes=INFERENCE_NUM_NODES)
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
