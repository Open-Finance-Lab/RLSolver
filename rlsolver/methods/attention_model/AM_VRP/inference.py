import torch
import numpy as np
import time
from tqdm import tqdm
from itertools import permutations

from config import Config
from model import AttentionDynamicModel
from environment import AgentVRP
from utils import generate_data_onfly


def nearest_neighbor_heuristic(depot, locations, demands, capacity=1.0):
    """Nearest neighbor heuristic for VRP."""
    batch_size = locations.size(0)
    n_locations = locations.size(1)
    device = locations.device
    
    all_costs = []
    
    for b in range(batch_size):
        depot_b = depot[b]
        locs_b = locations[b]
        demand_b = demands[b]
        
        # Calculate distance matrix
        all_locs = torch.cat([depot_b.unsqueeze(0), locs_b], dim=0)
        dist_matrix = torch.cdist(all_locs, all_locs, p=2)
        
        visited = torch.zeros(n_locations + 1, dtype=torch.bool, device=device)
        visited[0] = True  # Depot is "visited"
        current_loc = 0
        current_capacity = 0
        total_distance = 0
        
        while not visited[1:].all():
            # Find unvisited customers that fit in current capacity
            feasible = torch.zeros(n_locations + 1, dtype=torch.bool, device=device)
            for i in range(1, n_locations + 1):
                if not visited[i] and current_capacity + demand_b[i-1] <= capacity:
                    feasible[i] = True
            
            if not feasible.any():
                # Return to depot and start new route
                total_distance += dist_matrix[current_loc, 0].item()
                current_loc = 0
                current_capacity = 0
            else:
                # Go to nearest feasible customer
                distances = dist_matrix[current_loc].clone()
                distances[~feasible] = float('inf')
                next_loc = distances.argmin().item()
                
                total_distance += dist_matrix[current_loc, next_loc].item()
                visited[next_loc] = True
                current_capacity += demand_b[next_loc - 1].item()
                current_loc = next_loc
        
        # Return to depot at the end
        total_distance += dist_matrix[current_loc, 0].item()
        all_costs.append(total_distance)
    
    return torch.tensor(all_costs, device=device)


def savings_algorithm(depot, locations, demands, capacity=1.0):
    """Clarke-Wright Savings Algorithm for VRP - a strong heuristic often close to optimal."""
    batch_size = locations.size(0)
    device = locations.device
    all_costs = []
    
    for b in range(batch_size):
        depot_b = depot[b]
        locs_b = locations[b]
        demand_b = demands[b]
        n_loc = locs_b.size(0)
        
        # Calculate distance matrix
        all_locs = torch.cat([depot_b.unsqueeze(0), locs_b], dim=0)
        dist_matrix = torch.cdist(all_locs, all_locs, p=2)
        
        # Calculate savings matrix
        savings = []
        for i in range(1, n_loc + 1):
            for j in range(i + 1, n_loc + 1):
                saving = dist_matrix[0, i] + dist_matrix[0, j] - dist_matrix[i, j]
                savings.append((saving.item(), i, j))
        
        # Sort savings in descending order
        savings.sort(reverse=True)
        
        # Initialize routes (each customer in separate route)
        routes = [[i] for i in range(1, n_loc + 1)]
        route_demands = [demand_b[i-1].item() for i in range(1, n_loc + 1)]
        
        # Merge routes based on savings
        for saving_value, i, j in savings:
            # Find routes containing i and j
            route_i = None
            route_j = None
            for r_idx, route in enumerate(routes):
                if i in route:
                    route_i = r_idx
                if j in route:
                    route_j = r_idx
            
            if route_i is not None and route_j is not None and route_i != route_j:
                # Check if routes can be merged (capacity constraint)
                if route_demands[route_i] + route_demands[route_j] <= capacity:
                    # Check if i and j are at the ends of their routes
                    can_merge = False
                    new_route = None
                    
                    if routes[route_i][0] == i and routes[route_j][-1] == j:
                        new_route = routes[route_j] + routes[route_i]
                        can_merge = True
                    elif routes[route_i][-1] == i and routes[route_j][0] == j:
                        new_route = routes[route_i] + routes[route_j]
                        can_merge = True
                    elif routes[route_i][0] == i and routes[route_j][0] == j:
                        new_route = routes[route_j][::-1] + routes[route_i]
                        can_merge = True
                    elif routes[route_i][-1] == i and routes[route_j][-1] == j:
                        new_route = routes[route_i] + routes[route_j][::-1]
                        can_merge = True
                    
                    if can_merge and new_route:
                        # Merge routes
                        routes[route_i] = new_route
                        route_demands[route_i] += route_demands[route_j]
                        del routes[route_j]
                        del route_demands[route_j]
        
        # Calculate total cost
        total_cost = 0
        for route in routes:
            # Add depot -> first customer
            total_cost += dist_matrix[0, route[0]].item()
            # Add customer to customer distances
            for i in range(len(route) - 1):
                total_cost += dist_matrix[route[i], route[i+1]].item()
            # Add last customer -> depot
            total_cost += dist_matrix[route[-1], 0].item()
        
        all_costs.append(total_cost)
    
    return torch.tensor(all_costs, device=device)


def branch_and_bound_vrp(depot, locations, demands, capacity=1.0, max_nodes=10, time_limit=5.0):
    """
    Branch and Bound for small VRP instances.
    Only works well for very small instances (max_nodes <= 10).
    """
    batch_size = locations.size(0)
    device = locations.device
    all_costs = []
    
    for b in range(batch_size):
        depot_b = depot[b]
        locs_b = locations[b]
        demand_b = demands[b]
        n_loc = min(locs_b.size(0), max_nodes)  # Limit problem size
        
        if n_loc > max_nodes:
            # Use savings algorithm for larger instances
            cost = savings_algorithm(
                depot[b:b+1], 
                locations[b:b+1, :max_nodes], 
                demands[b:b+1, :max_nodes], 
                capacity
            )
            all_costs.append(cost.item())
            continue
        
        # Calculate distance matrix
        all_locs = torch.cat([depot_b.unsqueeze(0), locs_b[:n_loc]], dim=0)
        dist_matrix = torch.cdist(all_locs, all_locs, p=2).cpu().numpy()
        demands_np = demand_b[:n_loc].cpu().numpy()
        
        best_cost = float('inf')
        start_time = time.time()
        
        def calculate_route_cost(route):
            """Calculate cost of a route starting and ending at depot."""
            if not route:
                return 0
            cost = dist_matrix[0, route[0]]  # depot to first
            for i in range(len(route) - 1):
                cost += dist_matrix[route[i], route[i+1]]
            cost += dist_matrix[route[-1], 0]  # last to depot
            return cost
        
        def partition_into_routes(customers, capacity):
            """Partition customers into feasible routes."""
            if not customers:
                return [[]]
            
            routes = []
            current_route = []
            current_capacity = 0
            
            for c in customers:
                if current_capacity + demands_np[c-1] <= capacity:
                    current_route.append(c)
                    current_capacity += demands_np[c-1]
                else:
                    if current_route:
                        routes.append(current_route)
                    current_route = [c]
                    current_capacity = demands_np[c-1]
            
            if current_route:
                routes.append(current_route)
            
            return routes
        
        # Try different permutations (limited by time)
        customers = list(range(1, n_loc + 1))
        
        # For very small instances, try all permutations
        if n_loc <= 8:
            for perm in permutations(customers):
                if time.time() - start_time > time_limit:
                    break
                
                routes = partition_into_routes(perm, capacity)
                total_cost = sum(calculate_route_cost(route) for route in routes)
                best_cost = min(best_cost, total_cost)
        else:
            # For larger instances, use sampling
            for _ in range(min(5000, np.math.factorial(n_loc))):
                if time.time() - start_time > time_limit:
                    break
                
                perm = np.random.permutation(customers).tolist()
                routes = partition_into_routes(perm, capacity)
                total_cost = sum(calculate_route_cost(route) for route in routes)
                best_cost = min(best_cost, total_cost)
        
        all_costs.append(best_cost)
    
    return torch.tensor(all_costs, device=device)


def evaluate_model(model, dataset, mode='greedy'):
    """Evaluate trained model on dataset."""
    model.eval()
    model.set_decode_type(mode)
    
    costs_list = []
    times_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataset, desc=f"Evaluating model ({mode})"):
            batch = [b.to(model.device) for b in batch]
            
            start_time = time.time()
            cost, _ = model(batch)
            end_time = time.time()
            
            costs_list.append(cost)
            times_list.append(end_time - start_time)
    
    all_costs = torch.cat(costs_list, dim=0)
    avg_time = np.mean(times_list)
    
    return all_costs, avg_time


def main():
    """Main evaluation function."""
    config = Config()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load trained model
    print('Loading trained model...')
    model = AttentionDynamicModel(
        embedding_dim=config.embedding_dim,
        n_encode_layers=config.n_encode_layers,
        n_heads=config.n_heads,
        tanh_clipping=config.tanh_clipping
    ).to(device)
    
    try:
        state_dict = torch.load('model.pth', map_location=device)
        model.load_state_dict(state_dict)
        model.device = device
        print('Model loaded successfully')
    except FileNotFoundError:
        print('Warning: model.pth not found, using untrained model')
    
    # Generate test dataset
    test_size = 100  # Reduced for optimal solution computation
    print(f'\nGenerating test dataset with {test_size} instances...')
    test_dataset = generate_data_onfly(
        num_samples=test_size,
        graph_size=config.graph_size,
        device=device
    )
    
    # Collect all test data for heuristics
    all_depots = []
    all_locations = []
    all_demands = []
    for batch in test_dataset:
        all_depots.append(batch[0])
        all_locations.append(batch[1])
        all_demands.append(batch[2])
    
    depot = torch.cat(all_depots, dim=0)
    locations = torch.cat(all_locations, dim=0)
    demands = torch.cat(all_demands, dim=0)
    
    results = {}
    
    # 1. Evaluate trained model (greedy)
    print('\n1. Evaluating trained model (greedy)...')
    model_costs_greedy, model_time_greedy = evaluate_model(model, test_dataset, mode='greedy')
    results['Trained Model (Greedy)'] = {
        'mean': model_costs_greedy.mean().item(),
        'std': model_costs_greedy.std().item(),
        'min': model_costs_greedy.min().item(),
        'max': model_costs_greedy.max().item(),
        'time': model_time_greedy
    }
    
    # 2. Evaluate trained model (sampling)
    print('2. Evaluating trained model (sampling)...')
    model_costs_sampling, model_time_sampling = evaluate_model(model, test_dataset, mode='sampling')
    results['Trained Model (Sampling)'] = {
        'mean': model_costs_sampling.mean().item(),
        'std': model_costs_sampling.std().item(),
        'min': model_costs_sampling.min().item(),
        'max': model_costs_sampling.max().item(),
        'time': model_time_sampling
    }
    
    # 3. Nearest Neighbor Heuristic
    print('3. Evaluating Nearest Neighbor heuristic...')
    start_time = time.time()
    nn_costs = nearest_neighbor_heuristic(depot, locations, demands)
    nn_time = (time.time() - start_time) / test_size
    results['Nearest Neighbor'] = {
        'mean': nn_costs.mean().item(),
        'std': nn_costs.std().item(),
        'min': nn_costs.min().item(),
        'max': nn_costs.max().item(),
        'time': nn_time
    }
    
    # 4. Savings Algorithm (Strong Heuristic)
    print('4. Evaluating Savings Algorithm...')
    start_time = time.time()
    savings_costs = savings_algorithm(depot, locations, demands)
    savings_time = (time.time() - start_time) / test_size
    results['Savings Algorithm'] = {
        'mean': savings_costs.mean().item(),
        'std': savings_costs.std().item(),
        'min': savings_costs.min().item(),
        'max': savings_costs.max().item(),
        'time': savings_time
    }
    
    # 5. Near-Optimal Solution (for small instances)
    if config.graph_size <= 10:
        print('5. Computing near-optimal solutions (Branch & Bound)...')
        start_time = time.time()
        optimal_costs = branch_and_bound_vrp(depot, locations, demands, max_nodes=10)
        optimal_time = (time.time() - start_time) / test_size
        results['Near-Optimal (B&B)'] = {
            'mean': optimal_costs.mean().item(),
            'std': optimal_costs.std().item(),
            'min': optimal_costs.min().item(),
            'max': optimal_costs.max().item(),
            'time': optimal_time
        }
    else:
        print(f'5. Skipping exact solution (graph_size={config.graph_size} too large)')
    
    # Print results
    print('\n' + '='*80)
    print(f'PERFORMANCE COMPARISON (VRP with {config.graph_size} customers)')
    print('='*80)
    print(f'{"Algorithm":<30} {"Mean Cost":<12} {"Std":<10} {"Min":<10} {"Max":<10} {"Time(s)":<10}')
    print('-'*80)
    
    # Sort by mean cost
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean'])
    
    for name, metrics in sorted_results:
        print(f'{name:<30} {metrics["mean"]:<12.4f} {metrics["std"]:<10.4f} '
              f'{metrics["min"]:<10.4f} {metrics["max"]:<10.4f} {metrics["time"]:<10.6f}')
    
    # Calculate improvements
    print('\n' + '='*80)
    print('RELATIVE PERFORMANCE')
    print('='*80)
    
    # Use best algorithm as reference
    best_name, best_metrics = sorted_results[0]
    print(f'Best algorithm: {best_name} (mean cost: {best_metrics["mean"]:.4f})\n')
    
    for name, metrics in sorted_results:
        gap = (metrics['mean'] - best_metrics['mean']) / best_metrics['mean'] * 100
        print(f'{name:<30} Gap from best: {gap:+.2f}%')
    
    # Statistical significance test
    print('\n' + '='*80)
    print('STATISTICAL SIGNIFICANCE TEST')
    print('='*80)
    
    from scipy.stats import ttest_rel
    
    # Compare trained model with heuristics
    comparisons = [
        ('Trained Model (Greedy)', model_costs_greedy, 'Nearest Neighbor', nn_costs),
        ('Trained Model (Greedy)', model_costs_greedy, 'Savings Algorithm', savings_costs),
    ]
    
    if config.graph_size <= 20 and 'Near-Optimal (B&B)' in results:
        comparisons.append(
            ('Trained Model (Greedy)', model_costs_greedy, 'Near-Optimal (B&B)', optimal_costs)
        )
    
    for name1, costs1, name2, costs2 in comparisons:
        t_stat, p_value = ttest_rel(costs1.cpu().numpy(), costs2.cpu().numpy())
        if p_value < 0.05:
            better = name1 if costs1.mean() < costs2.mean() else name2
            print(f'{name1} vs {name2}: p={p_value:.4f} - {better} is significantly better')
        else:
            print(f'{name1} vs {name2}: p={p_value:.4f} - No significant difference')


if __name__ == '__main__':
    main()