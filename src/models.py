"""Per-point MLP and GraphSAGE surrogate models."""
import numpy as np
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
        """k-NN graph via scipy cKDTree instead of torch_geometric.nn.knn_graph,
        which requires the pyg-lib/torch-cluster compiled backend (not part of
        the fresh-Mac install list and not trivially pip-installable on Apple
        Silicon). cKDTree runs on CPU regardless of pos's device -- the
        BUILD_SPEC already expects some scatter/graph ops to fall back off MPS."""
        from scipy.spatial import cKDTree

        pos_np = pos.detach().cpu().numpy()
        tree = cKDTree(pos_np)
        _, idx = tree.query(pos_np, k=self.k + 1)  # includes self at k=0
        idx = idx[:, 1:]  # drop self
        n = pos_np.shape[0]
        src = np.repeat(np.arange(n), self.k)
        dst = idx.reshape(-1)
        edge_index = np.stack([dst, src], axis=0)  # message flows neighbor -> center
        return torch.from_numpy(edge_index).long().to(pos.device)


def build_model(name, **kwargs):
    if name == "mlp":
        return PerPointMLP(**kwargs)
    if name == "graphsage":
        return GraphSAGENet(**kwargs)
    raise ValueError(f"unknown model {name}")
