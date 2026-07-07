"""Per-point MLP and GraphSAGE surrogate models."""
import torch
import torch.nn as nn


class PerPointMLP(nn.Module):
    def __init__(self, in_dim=7, hidden=(128, 128, 128, 128), out_dim=4):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, edge_index=None):
        return self.net(x)


class GraphSAGENet(nn.Module):
    def __init__(self, in_dim=7, hidden=128, out_dim=4, n_layers=4, k=10):
        super().__init__()
        from torch_geometric.nn import SAGEConv

        self.k = k
        self.convs = nn.ModuleList()
        prev = in_dim
        for _ in range(n_layers):
            self.convs.append(SAGEConv(prev, hidden))
            prev = hidden
        self.out = nn.Linear(hidden, out_dim)
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        h = x
        for conv in self.convs:
            h = self.act(conv(h, edge_index))
        return self.out(h)

    def build_graph(self, pos):
        from torch_geometric.nn import knn_graph

        return knn_graph(pos, k=self.k, loop=False)


def build_model(name, **kwargs):
    if name == "mlp":
        return PerPointMLP(**kwargs)
    if name == "graphsage":
        return GraphSAGENet(**kwargs)
    raise ValueError(f"unknown model {name}")
