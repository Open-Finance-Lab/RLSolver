import os
import random
import argparse
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
from rlsolver.methods.PIGNN.util import eval_maxcut, eval_MIS
from rlsolver.methods.PIGNN.config import *

def run(args):
    # Seed
    os.environ["PL_GLOBAL_SEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Create datasets and loaders
    #  in_dim = NUM_NODES ** 0.5 if aNUM_NODES >= 1e5 else NUM_NODES ** (1/3) # In the example code provided by the authors they don't use the cubic root, even though it is stated in the paper
    in_dim = NUM_NODES ** 0.5
    in_dim = round(in_dim)
    dataset = DRegDataset(NODE_DEGREE, NUM_GRAPHS, NUM_NODES, in_dim, SEED)
    print('dataset len:', len(dataset))
    dataloader = DataLoader(dataset.data, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS)
    print('dataloader ready...')

    # Build model
    hidden_dim = round(in_dim/2)
    model = PIGNN(
        in_dim, 
        hidden_dim, 
        PROBLEM,
        lr=LEARNING_RATE,
        out_dim=1, 
        num_heads=NUM_HEADS,
        layer_type=GNN_MODEL
    )

    # Training (via PyL)
    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=1e-4, 
        patience=1000, 
        verbose=True, 
        mode="min"
    )

    trainer = Trainer(
        callbacks=[early_stop_callback],
        devices=[0],
        accelerator='gpu',
        max_epochs=EPOCHS,
        check_val_every_n_epoch=1,
    )

    start_time = time()
    trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=dataloader)

    # Evaluate after training
    model.eval()
    eval_fn = eval_maxcut if PROBLEM == Problem.maxcut else eval_MIS

    with torch.no_grad():
        e, a = [], []
        for batch in tqdm(dataloader, desc='evaluating model...'):
            x, edge_index = batch.x, batch.edge_index
            pred = model(x, edge_index)
            proj = torch.round(pred)
            energy, approx_ratio = eval_fn(edge_index, proj, NODE_DEGREE, NUM_GRAPHS)
            e.append(energy.item()), a.append(approx_ratio.item())

    print(f'Avg. estimated energy: {np.mean(e)}, avg. approximation ratio: {np.mean(a)}')    
    print(f'Completed training and evaluation for seed={SEED} in {round(time()-start_time, 2)}s')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_graphs', type=int, default=100)
    parser.add_argument('--num_nodes', type=int, default=100)
    parser.add_argument('--node_degree', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=int(1e5))
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--maxcut', action='store_true', help='If this flag is true solve the maxcut problem, else solve mis')
    parser.add_argument('--gnn_model', type=int, default=0)
    parser.add_argument('--num_heads', type=int, default=4, help='Nr of heads if you wish to use GAT Ansatz')

    args = parser.parse_args()
    run(args)


