import torch as th
import torch.nn as nn

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

class Net(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding_layer = nn.Embedding(num_nodes, embedding_dim)
        nn.init.orthogonal_(self.embedding_layer.weight)

    def forward(self, ids):
        embedded_x = self.embedding_layer(ids)
        return embedded_x

class OptNet(nn.Module):
    def __init__(self, N, hidden_layers):
        super(OptNet, self).__init__()
        self.N = N
        self.hidden_layers = hidden_layers
        self.lstm = nn.LSTM(self.N, self.hidden_layers, 1, batch_first=True)
        self.output = nn.Linear(hidden_layers, self.N)

    def forward(self, configuration, hidden_state, cell_state):
        x, (h, c) = self.lstm(configuration, (hidden_state, cell_state))
        return self.output(x).sigmoid(), h, c

class PolicyMLP(nn.Module):
    def __init__(self, node_dim, mid_dim, embedding_dim):
        super().__init__()
        self.net1 = nn.Sequential(nn.Linear(node_dim, mid_dim), nn.GELU(), nn.LayerNorm(mid_dim),
                                  nn.Linear(mid_dim, mid_dim), nn.GELU(), nn.LayerNorm(mid_dim), )
        self.net2 = nn.Sequential(nn.Linear(mid_dim + embedding_dim, mid_dim), nn.GELU(), nn.LayerNorm(mid_dim),
                                  nn.Linear(mid_dim, 1), nn.Sigmoid(), )

    def forward(self, solutions, embedding_ws):
        solutions1 = self.net1(solutions)
        solutions2 = self.net2(th.concat((solutions1, embedding_ws), dim=1))
        return solutions2.squeeze(1)

class PolicyGNN(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim):
        super().__init__()
        self.inp_dim = inp_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim

        self.inp_enc = nn.Sequential(nn.Linear(inp_dim, mid_dim), nn.ReLU(), nn.LayerNorm(mid_dim),
                                     nn.Linear(mid_dim, mid_dim))

        self.tmp_enc = nn.Sequential(nn.Linear(mid_dim + mid_dim, mid_dim), nn.ReLU(), nn.LayerNorm(mid_dim),
                                     nn.Linear(mid_dim, out_dim), )
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, inp, ids_list):
        num_size, num_nodes, num_dim = inp.shape
        device = inp.device

        tmp1 = th.empty((num_size, num_nodes, self.mid_dim), dtype=th.float32, device=device)
        for node0 in range(num_nodes):
            tmp1[:, node0] = self.inp_enc(inp[:, node0])

        env_i = th.arange(inp.shape[0], device=inp.device)
        tmp2 = th.empty((num_size, num_nodes, self.mid_dim), dtype=th.float32, device=device)
        for node0, node1s in enumerate(ids_list):
            tmp2[:, node0, :] = tmp1[env_i[:, None], node1s[None, :]].mean(dim=1)

        tmp3 = th.cat((tmp1, tmp2), dim=2)
        out = th.empty((num_size, num_nodes, self.out_dim), dtype=th.float32, device=device)
        for node0 in range(num_nodes):
            out[:, node0] = self.tmp_enc(tmp3[:, node0])
        return self.soft_max(out.squeeze(2))

class OptimizerLSTM(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, num_layers):
        super().__init__()
        self.inp_dim = inp_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        self.rnn0 = nn.LSTM(inp_dim, mid_dim, num_layers=num_layers)
        self.mlp0 = nn.Linear(mid_dim, out_dim)
        self.rnn1 = nn.LSTM(1, 8, num_layers=num_layers)
        self.mlp1 = nn.Linear(8, 1)

    def forward(self, inp, hid0=None, hid1=None):
        tmp0, hid0 = self.rnn0(inp, hid0)
        out0 = self.mlp0(tmp0)

        d0, d1, d2 = inp.shape
        inp1 = inp.reshape(d0, d1 * d2, 1)
        tmp1, hid1 = self.rnn1(inp1, hid1)
        out1 = self.mlp1(tmp1).reshape(d0, d1, d2)

        out = out0 + out1
        return out, hid0, hid1


class OptimizerLSTM2(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, num_layers):
        super().__init__()
        inp_dim0 = inp_dim[0]
        inp_dim1 = inp_dim[1]

        self.rnn0 = nn.LSTM(inp_dim0, mid_dim, num_layers=num_layers)
        self.enc0 = nn.Sequential(nn.Linear(inp_dim1, mid_dim), nn.ReLU(),
                                  nn.Linear(mid_dim, mid_dim))  # todo graph dist
        self.mlp0 = nn.Sequential(nn.Linear(mid_dim + mid_dim, mid_dim), nn.ReLU(),
                                  nn.Linear(mid_dim, out_dim))

        self.rnn1 = nn.LSTM(1, 8, num_layers=num_layers)
        self.mlp1 = nn.Linear(8, 1)

    def forward(self, inp, graph, hid0=None, hid1=None):
        d0, d1, d2 = inp.shape
        inp1 = inp.reshape(d0, d1 * d2, 1)
        tmp1, hid1 = self.rnn1(inp1, hid1)
        out1 = self.mlp1(tmp1).reshape(d0, d1, d2)

        tmp0, hid0 = self.rnn0(inp, hid0)

        graph = self.enc0(graph) * th.ones((d0, d1, 1), device=tmp0.device)
        tmp0 = th.concat((tmp0, graph), dim=2)  # todo graph dist
        out0 = self.mlp0(tmp0)

        out = out0 + out1
        return out, hid0, hid1