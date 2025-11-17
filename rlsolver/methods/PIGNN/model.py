import torch
import pytorch_lightning as L
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, GraphConv
from rlsolver.methods.PIGNN.util import hamiltonian_maxcut, hamiltonian_MIS, hamiltonian_graph_coloring


from rlsolver.methods.PIGNN.config import *
class PIGNN(L.LightningModule):
    def __init__(self, in_dim, hidden_dim, problem, lr=1e-3, out_dim=1, num_heads=1, layer_type=MODEL_KEY_DICT.GCN):
        """
        The base class of a gnn classifier. The layer type indicates which gnn to use.
        The different options are:
        0 - GCN
        1 - GAT
        2 - GATv2
        3 - GraphConv
        """
        super(PIGNN, self).__init__()
        self.layer_type = layer_type
        self.lr = lr
        self.problem = problem  # Store problem type for forward pass

        # Select appropriate Hamiltonian loss function based on problem type
        if problem == Problem.maxcut:
            self.loss_fn = hamiltonian_maxcut
        elif problem == Problem.MIS:
            self.loss_fn = hamiltonian_MIS
        elif problem == Problem.graph_coloring:
            # For graph coloring, we need to pass the additional parameters
            def graph_coloring_loss_with_params(edge_index, pred):
                return hamiltonian_graph_coloring(edge_index, pred, LAMBDA_ENTROPY, LAMBDA_BALANCE)
            self.loss_fn = graph_coloring_loss_with_params
        else:
            raise ValueError(f"Unsupported problem type: {problem}")
        
        if layer_type == MODEL_KEY_DICT.GCN:
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, out_dim)
        elif layer_type == MODEL_KEY_DICT.GAT:
            self.conv1 = GATConv(in_dim, hidden_dim, heads=num_heads)
            self.conv2 = GATConv(hidden_dim*num_heads, out_dim, heads=num_heads, concat=False)
        elif layer_type == MODEL_KEY_DICT.GATv2:
            self.conv1 = GATv2Conv(in_dim, hidden_dim, heads=num_heads)
            self.conv2 = GATv2Conv(hidden_dim*num_heads, out_dim, heads=num_heads, concat=False)
        elif layer_type == MODEL_KEY_DICT.GraphConv:
            self.conv1 = GraphConv(in_dim, hidden_dim)
            self.conv2 = GraphConv(hidden_dim, out_dim) 
        self.save_hyperparameters()


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        logits = self.conv2(x, edge_index)

        # Apply different activation functions based on problem type
        if self.problem == Problem.graph_coloring:
            # Multi-class classification for graph coloring
            return torch.softmax(logits, dim=1)
        else:
            # Binary classification for maxcut and MIS
            return torch.sigmoid(logits)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        return [optimizer]
    
    def training_step(self, batch, batch_idx):
        x, edge_index = batch.x, batch.edge_index
        pred = self.forward(x, edge_index)
        loss = self.loss_fn(edge_index, pred)
        self.log('train_loss', loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, edge_index = batch.x, batch.edge_index
        pred = self.forward(x, edge_index)

        # For maxcut and MIS, round predictions to binary values for validation
        # For graph coloring, use softmax probabilities directly
        if self.problem == Problem.graph_coloring:
            loss = self.loss_fn(edge_index, pred)
        else:
            loss = self.loss_fn(edge_index, pred.round())  # round to nearest int for validation, i.e. calc QUBO
        self.log('val_loss', loss, prog_bar=True)