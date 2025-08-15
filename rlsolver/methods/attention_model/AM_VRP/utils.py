import torch
import torch.utils.data as data
import torch.distributed as dist
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


CAPACITIES = {
    10: 20.,
    20: 30.,
    50: 40.,
    100: 50.
}


def generate_data_onfly_distributed(num_samples, graph_size, batch_size, rank, world_size, device='cuda'):
    """Generate VRP dataset distributed across GPUs."""
    # Calculate local samples for this rank
    local_samples = num_samples // world_size
    if rank < num_samples % world_size:
        local_samples += 1
    
    # Generate local data
    depot = torch.rand(local_samples, 2, device=device)
    loc = torch.rand(local_samples, graph_size, 2, device=device)
    demand = torch.randint(1, 10, (local_samples, graph_size), device=device).float()
    demand = demand / CAPACITIES[graph_size]
    
    dataset = data.TensorDataset(depot, loc, demand)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def generate_data_onfly(num_samples=10000, graph_size=20, device='cuda'):
    """Generate VRP dataset on the fly (non-distributed version)."""
    depot = torch.rand(num_samples, 2, device=device)
    loc = torch.rand(num_samples, graph_size, 2, device=device)
    demand = torch.randint(1, 10, (num_samples, graph_size), device=device).float()
    demand = demand / CAPACITIES[graph_size]
    
    dataset = data.TensorDataset(depot, loc, demand)
    return data.DataLoader(dataset, batch_size=128, shuffle=False)


def create_data_on_disk(graph_size, num_samples, filename=None, seed=1234, rank=0):
    """Generate and save validation dataset."""
    torch.manual_seed(seed)
    
    depot = torch.rand(num_samples, 2)
    loc = torch.rand(num_samples, graph_size, 2)
    demand = torch.randint(1, 10, (num_samples, graph_size)).float()
    demand = demand / CAPACITIES[graph_size]
    
    if filename and rank == 0:
        with open(f'validation_dataset_{filename}.pkl', 'wb') as f:
            pickle.dump((depot, loc, demand), f)
    
    dataset = data.TensorDataset(depot, loc, demand)
    return data.DataLoader(dataset, batch_size=128, shuffle=False)


def read_from_pickle(path, batch_size=128, num_samples=None, device='cuda'):
    """Load dataset from pickle file."""
    with open(path, 'rb') as f:
        depot, loc, demand = pickle.load(f)
    
    if num_samples is not None:
        depot = depot[:num_samples]
        loc = loc[:num_samples]
        demand = demand[:num_samples]
    
    depot = depot.to(device)
    loc = loc.to(device)
    demand = demand.to(device)
    
    dataset = data.TensorDataset(depot, loc, demand)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def get_results(train_loss, train_cost, val_cost, save_results=True, filename=None, plots=True):
    """Save and plot training results."""
    epochs_num = len(train_loss)
    
    df_train = pd.DataFrame({
        'epochs': list(range(epochs_num)),
        'loss': train_loss,
        'cost': train_cost
    })
    
    df_test = pd.DataFrame({
        'epochs': list(range(epochs_num)),
        'val_cost': val_cost
    })
    
    if save_results and filename:
        df_train.to_excel(f'train_results_{filename}.xlsx', index=False)
        df_test.to_excel(f'test_results_{filename}.xlsx', index=False)
    
    if plots:
        plt.figure(figsize=(12, 6))
        ax = sns.lineplot(x='epochs', y='loss', data=df_train, color='salmon', label='train loss')
        ax2 = ax.twinx()
        sns.lineplot(x='epochs', y='cost', data=df_train, color='cornflowerblue', 
                    label='train cost', ax=ax2)
        sns.lineplot(x='epochs', y='val_cost', data=df_test, color='darkblue', 
                    label='val cost', ax=ax2)
        
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax2.set_ylabel('Cost')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        if filename:
            plt.savefig(f'learning_curve_{filename}.png', dpi=100, bbox_inches='tight')
        plt.show()


def get_clean_path(path):
    """Remove duplicate zeros from path."""
    cleaned = []
    prev = None
    
    for node in path:
        if node != prev or node != 0:
            cleaned.append(node)
        prev = node
    
    if cleaned[0] != 0:
        cleaned.insert(0, 0)
    if cleaned[-1] != 0:
        cleaned.append(0)
    
    return cleaned


def plot_vrp_solution(depot, locations, tour, demands=None):
    """Visualize VRP solution."""
    plt.figure(figsize=(10, 10))
    
    plt.scatter(depot[0], depot[1], c='red', s=200, marker='s', label='Depot')
    plt.scatter(locations[:, 0], locations[:, 1], c='blue', s=100, label='Customers')
    
    if demands is not None:
        for i, (x, y, d) in enumerate(zip(locations[:, 0], locations[:, 1], demands)):
            plt.annotate(f'{i+1}\n({d:.2f})', (x, y), fontsize=8, ha='center')
    
    tour_clean = get_clean_path(tour)
    all_locs = torch.cat([depot.unsqueeze(0), locations])
    
    for i in range(len(tour_clean) - 1):
        start = all_locs[int(tour_clean[i])]
        end = all_locs[int(tour_clean[i + 1])]
        plt.plot([start[0], end[0]], [start[1], end[1]], 'g-', alpha=0.6)
    
    plt.legend()
    plt.title('VRP Solution')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.grid(True, alpha=0.3)
    plt.show()


def get_cur_time():
    """Get current time as string."""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')