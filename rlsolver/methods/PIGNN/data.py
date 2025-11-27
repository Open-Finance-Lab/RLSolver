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