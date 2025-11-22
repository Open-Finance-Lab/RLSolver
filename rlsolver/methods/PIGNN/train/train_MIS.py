"""
Train script for MIS (Maximum Independent Set) problem using PIGNN (Physics-Inspired Graph Neural Network).

This script implements the training pipeline for the MIS optimization problem
using PyTorch Lightning and PyTorch Geometric frameworks.
"""

import os
import random
import torch
import numpy as np
from tqdm import tqdm
from time import time
from torch_geometric.loader import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Use new unified environment structure
from rlsolver.methods.PIGNN.data import DRegDataset
from rlsolver.methods.PIGNN.model import PIGNN
from rlsolver.methods.PIGNN.util import eval_MIS
from rlsolver.methods.PIGNN.config import *
from rlsolver.methods.config import Problem


def run():
    """
    Main training function for MIS problem.

    This function implements the complete training pipeline:
    1. Environment setup and seeding
    2. Dataset creation and loading
    3. Model initialization
    4. Training configuration
    5. Training execution
    6. Evaluation and results reporting
    """
    # Set problem type explicitly to MIS
    globals()['PROBLEM'] = Problem.MIS

    # Seed setup for reproducibility
    os.environ["PL_GLOBAL_SEED"] = str(TRAIN_SEED)
    random.seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)
    torch.cuda.manual_seed_all(TRAIN_SEED)

    print(f"Starting MIS training with {TRAIN_NUM_GRAPHS} graphs, {TRAIN_NUM_NODES} nodes each")
    print(f"Training configuration: {TRAIN_EPOCHS} epochs, LR={TRAIN_LEARNING_RATE}")

    # Create datasets and loaders
    # Calculate feature dimension based on problem size
    in_dim = TRAIN_NUM_NODES ** 0.5
    in_dim = round(in_dim)

    dataset = DRegDataset(NODE_DEGREE, TRAIN_NUM_GRAPHS, TRAIN_NUM_NODES, in_dim, TRAIN_SEED)
    print(f'Dataset created: {len(dataset)} graphs')

    dataloader = DataLoader(
        dataset.data,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False,
        num_workers=TRAIN_NUM_WORKERS
    )
    print('Dataloader ready...')

    # Build PIGNN model for MIS
    hidden_dim = round(in_dim / 2)
    model = PIGNN(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        problem=PROBLEM,
        lr=TRAIN_LEARNING_RATE,
        out_dim=1,
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
            x, edge_index = batch.x, batch.edge_index
            pred = model(x, edge_index)
            proj = torch.round(pred)

            # Evaluate using MIS-specific evaluation function
            energy, approx_ratio = eval_MIS(edge_index, proj, NODE_DEGREE, TRAIN_NUM_GRAPHS)
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
        model_save_path = get_model_save_path(problem="MIS", num_nodes=TRAIN_NUM_NODES)
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to: {model_save_path}")


if __name__ == '__main__':
    run()
