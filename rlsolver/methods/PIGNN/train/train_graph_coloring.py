"""
Train script for Graph Coloring problem using PIGNN (Physics-Inspired Graph Neural Network).

This script implements the training pipeline for the Graph Coloring optimization problem
using PyTorch Lightning and PyTorch Geometric frameworks.
"""

import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Use unified environment structure
from rlsolver.methods.PIGNN.data import DRegDataset
from rlsolver.methods.PIGNN.model import PIGNN
from rlsolver.methods.PIGNN.util import eval_graph_coloring
from rlsolver.methods.PIGNN.config import *
from rlsolver.methods.config import Problem

# Import Graph Coloring environment from unified env file
from rlsolver.envs.env_PIGNN import PIGNNGraphColoringEnv

# Set up matplotlib for font support
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10


def temperature_sampling_decode(pred, edge_index, num_colors, temperature=TEMPERATURE, trials=TRIALS):
    """Temperature sampling decoding for a single graph."""
    device = pred.device
    num_nodes = pred.size(0)

    best_colors = None
    best_violations = float('inf')
    best_used_colors = float('inf')

    # Generate probability distribution for colors
    # Use sigmoid activation and scale to [0, num_colors-1]
    color_probs = torch.sigmoid(pred)

    for trial in range(trials):
        # Temperature sampling
        if temperature > 0:
            # Apply temperature and sample
            scaled_probs = color_probs / temperature
            uniform_probs = torch.rand(num_nodes, num_colors, device=device)
            colors = (scaled_probs.unsqueeze(-1) > uniform_probs).sum(dim=-1)
        else:
            # Greedy assignment
            colors = (color_probs > 0.5).long().squeeze()

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


def _split_batch_predictions(batch, predictions: torch.Tensor):
    batch = _to_batch(batch)
    graphs = batch.to_data_list()
    pointer = 0

    for data in graphs:
        data = data.to(predictions.device)
        num_nodes = data.num_nodes
        graph_pred = predictions[pointer:pointer + num_nodes]
        pointer += num_nodes
        yield data, graph_pred


def run():
    """
    Main training function for Graph Coloring problem.

    This function implements the complete training pipeline:
    1. Environment setup and seeding
    2. Dataset creation and loading
    3. Model initialization
    4. Training configuration
    5. Training execution
    6. Evaluation and results reporting
    """
    # Set problem type explicitly to Graph Coloring
    globals()['PROBLEM'] = Problem.graph_coloring

    # Seed setup for reproducibility
    os.environ["PL_GLOBAL_SEED"] = str(TRAIN_SEED)
    random.seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)
    torch.cuda.manual_seed_all(TRAIN_SEED)

    print(f"Starting Graph Coloring training with {TRAIN_NUM_GRAPHS} graphs, {TRAIN_NUM_NODES} nodes each")
    print(f"Training configuration: {TRAIN_EPOCHS} epochs, LR={TRAIN_LEARNING_RATE}")
    print(f"Graph Coloring specific: {NUM_COLORS} colors, temp={TEMPERATURE}, trials={TRIALS}")

    # Create datasets and loaders using the same pattern as other problems
    # Note: Using DRegDataset for consistency, but can be replaced with GraphColoringDataset
    in_dim = IN_DIM  # Use specific feature dimension for graph coloring

    dataset = DRegDataset(NODE_DEGREE, TRAIN_NUM_GRAPHS, TRAIN_NUM_NODES, in_dim, TRAIN_SEED)
    print(f'Dataset created: {len(dataset)} graphs')

    dataloader = DataLoader(
        dataset.data,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False,
        num_workers=TRAIN_NUM_WORKERS
    )
    print('Dataloader ready...')

    # Build PIGNN model for Graph Coloring
    hidden_dim = round(in_dim / 2)
    model = PIGNN(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        problem=PROBLEM,
        lr=TRAIN_LEARNING_RATE,
        out_dim=NUM_COLORS,  # Multi-color output
        num_heads=NUM_HEADS,
        layer_type=GNN_MODEL
    )

    print(f'Model initialized: {PROBLEM.name} problem, {GNN_MODEL.value} architecture')

    # Configure early stopping callback
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=1000,
        verbose=True,
        mode="min"
    )

    # Configure device based on GPU setting
    device_list = [TRAIN_GPU_NUM] if TRAIN_GPU_NUM >= 0 else 'auto'
    accelerator = 'gpu' if TRAIN_GPU_NUM >= 0 else 'cpu'

    # Configure PyTorch Lightning trainer
    trainer = Trainer(
        callbacks=[early_stop_callback],
        devices=device_list,
        accelerator=accelerator,
        max_epochs=TRAIN_EPOCHS,
        check_val_every_n_epoch=1,
    )

    print(f'Trainer configured: device={accelerator}, max_epochs={TRAIN_EPOCHS}')

    # Start training
    start_time = time()
    print("Starting training...")

    trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=dataloader)

    training_time = time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Evaluate after training
    print("Starting evaluation...")
    model.eval()

    with torch.no_grad():
        energies, approximation_ratios = [], []

        for batch in tqdm(dataloader, desc='Evaluating model...'):
            batch = batch.to(model.device) if hasattr(model, 'device') else batch
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

    # Report results
    avg_energy = np.mean(energies)
    avg_approx_ratio = np.mean(approximation_ratios)

    print(f'Evaluation Results:')
    print(f'  Average estimated energy: {avg_energy:.4f}')
    print(f'  Average approximation ratio: {avg_approx_ratio:.4f}')
    print(f'Training and evaluation completed for seed={TRAIN_SEED}')
    print(f'Total time: {round(time() - start_time, 2)}s')

    # Save model if needed
    if SAVE_MODEL:
        from rlsolver.methods.PIGNN.config import get_model_save_path
        model_save_path = get_model_save_path(problem="graph_coloring", num_nodes=TRAIN_NUM_NODES)
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to: {model_save_path}")


if __name__ == '__main__':
    run()
