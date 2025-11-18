import torch
import networkx as nx
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.utils import from_networkx

class DRegDataset(Dataset):
    def __init__(self, d=3, num_graphs=1000, num_nodes=100, in_dim=1, seed=0):
        super(DRegDataset, self).__init__()

        self.d = d
        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self.seed = seed
        self.in_dim = in_dim
        self.data = self.generate_data()

    def generate_data(self):
        data_list = []

        for _ in tqdm(range(self.num_graphs), desc=f'generating {self.num_graphs} random d-reg graphs...'):
            # Generate a random d-regular graph and append it to the data list
            g = nx.random_regular_graph(d=self.d, n=self.num_nodes, seed=self.seed)
            pyg = from_networkx(g)
            pyg.x = torch.randn(self.num_nodes, self.in_dim) # Not sure about this, might not be the best idea as init...
            data_list.append(pyg)

        return data_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

class GraphColoringDataset(Dataset):
    """
    Dataset for graph coloring problems.
    Generates various types of graphs that are suitable for coloring.
    """
    def __init__(self, num_graphs=100, num_nodes=25, num_colors=6, in_dim=16, seed=42):
        super(GraphColoringDataset, self).__init__()

        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self.num_colors = num_colors
        self.in_dim = in_dim
        self.seed = seed
        self.data = self.generate_data()

    def generate_data(self):
        """Generate graphs suitable for coloring with symmetry-breaking features."""
        import numpy as np

        data_list = []
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        print(f"Creating {self.num_graphs} graphs for graph coloring...")

        for i in tqdm(range(self.num_graphs)):
            # Generate different types of graphs for diversity
            if i % 3 == 0:
                # Erdős-Rényi random graph
                g = nx.erdos_renyi_graph(self.num_nodes, 0.25)
            elif i % 3 == 1:
                # Watts-Strogatz small-world graph
                g = nx.watts_strogatz_graph(self.num_nodes, 4, 0.3)
            else:
                # Barabási-Albert scale-free graph
                g = nx.barabasi_albert_graph(self.num_nodes, 3)

            # Ensure graph is connected
            if not nx.is_connected(g):
                components = list(nx.connected_components(g))
                for j in range(1, len(components)):
                    u = list(components[0])[0]
                    v = list(components[j])[0]
                    g.add_edge(u, v)

            # Convert to PyTorch Geometric format
            pyg = from_networkx(g)

            # 🔑 Symmetry-breaking input features (from user's proven implementation)
            # Random features
            random_features = torch.rand(self.num_nodes, self.in_dim // 2)

            # Node ID features to break symmetry
            node_id_features = torch.arange(self.num_nodes, dtype=torch.float32).unsqueeze(1) / self.num_nodes

            # Degree features normalized by max degree
            degree_features = torch.tensor([len(list(g.neighbors(j))) for j in range(self.num_nodes)],
                                        dtype=torch.float32).unsqueeze(1) / self.num_nodes

            # Additional random features to reach desired dimension
            remaining_dim = self.in_dim - random_features.shape[1] - node_id_features.shape[1] - degree_features.shape[1]
            if remaining_dim > 0:
                additional_features = torch.rand(self.num_nodes, remaining_dim)
            else:
                additional_features = torch.empty(self.num_nodes, 0)

            # Concatenate all features
            features = torch.cat([random_features, node_id_features, degree_features, additional_features], dim=1)

            # Normalize features
            features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-6)

            pyg.x = features
            pyg.num_colors = self.num_colors  # Store number of colors for reference

            data_list.append(pyg)

        return data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]