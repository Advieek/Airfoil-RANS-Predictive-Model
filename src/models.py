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
    def __init__(self, in_dim=7, hidden=128, out_dim=4, n_layers=4, k=10, knn_backend="mps_chunked", knn_chunk=4096):
        super().__init__()
        from torch_geometric.nn import SAGEConv

        self.k = k
        self.knn_backend = knn_backend
        self.knn_chunk = knn_chunk
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
        if self.knn_backend == "cpu_kdtree":
            return _knn_graph_cpu_kdtree(pos, self.k)
        try:
            return _knn_graph_gpu_chunked(pos, self.k, self.knn_chunk)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            # MPS memory headroom on this machine fluctuates with other apps'
            # GPU usage (WindowServer, browser compositing, ...) -- degrade
            # gracefully to the CPU backend for this call rather than crash an
            # unattended multi-hour run.
            torch.mps.empty_cache()
            return _knn_graph_cpu_kdtree(pos, self.k)


def _knn_graph_cpu_kdtree(pos, k):
    """k-NN graph via scipy cKDTree. CPU-bound regardless of pos's device --
    this was the original implementation and is kept as a fallback (e.g. if
    MPS lacks an op the chunked path needs)."""
    from scipy.spatial import cKDTree

    pos_np = pos.detach().cpu().numpy()
    tree = cKDTree(pos_np)
    _, idx = tree.query(pos_np, k=k + 1)  # includes self at k=0
    idx = idx[:, 1:]
    n = pos_np.shape[0]
    src = np.repeat(np.arange(n), k)
    dst = idx.reshape(-1)
    edge_index = np.stack([dst, src], axis=0)
    return torch.from_numpy(edge_index).long().to(pos.device)


def _knn_graph_gpu_chunked(pos, k, chunk=4096):
    """Brute-force k-NN done as chunked torch.cdist + topk, entirely on pos's
    own device (MPS on this machine). Replaces the cKDTree backend, which was
    the dominant per-epoch cost (~22s/epoch vs the MLP's ~1.7s/epoch) once
    profiled -- cKDTree is CPU-only and serial per simulation, while this
    leans on the GPU's matmul throughput instead. Chunked over query points so
    peak memory stays bounded even at full mesh resolution (~180k points/sim):
    a chunk-by-N distance matrix, not the full N-by-N one."""
    n = pos.shape[0]
    device = pos.device
    all_idx = torch.empty((n, k), dtype=torch.long, device=device)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        d = torch.cdist(pos[start:end], pos)  # (chunk, n)
        row_ids = torch.arange(start, end, device=device)
        d[torch.arange(end - start, device=device), row_ids] = float("inf")  # exclude self
        _, idx = torch.topk(d, k, dim=1, largest=False)
        all_idx[start:end] = idx
    src = torch.arange(n, device=device).repeat_interleave(k)
    dst = all_idx.reshape(-1)
    return torch.stack([dst, src], dim=0)


def build_batched_graph(model, pos_list):
    """Block-diagonal k-NN graph over multiple point clouds (one per
    simulation in a multi-sim training batch): each sim gets its own
    self-contained k-NN graph (no cross-simulation edges), concatenated with
    node-index offsets -- the standard PyG batching trick, done manually here
    since we're not using a PyG DataLoader."""
    edge_indices = []
    offset = 0
    for pos in pos_list:
        ei = model.build_graph(pos)
        edge_indices.append(ei + offset)
        offset += pos.shape[0]
    return torch.cat(edge_indices, dim=1)


def build_model(name, **kwargs):
    if name == "mlp":
        return PerPointMLP(**kwargs)
    if name == "graphsage":
        return GraphSAGENet(**kwargs)
    raise ValueError(f"unknown model {name}")
