"""Gate 3 check: real data pipeline. Loads scarce train split, splits train/val,
computes train-only norm stats, verifies a normalized DataLoader batch has
mean~0/std~1 and loads in <1s (cached path), and saves a sanity pressure plot.
"""
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from src.dataset import (
    SimPointCloudDataset,
    collate_single,
    compute_norm_stats,
    load_scarce_cached,
    save_norm_stats,
    split_train_val,
)

data_list, name_list = load_scarce_cached(root="data/Dataset", train=True)
print(f"n sims: {len(data_list)}, shape[0]: {data_list[0].shape}, dtype: {data_list[0].dtype}")

(train_data, train_names), (val_data, val_names) = split_train_val(data_list, name_list, n_val=20, seed=0)
print(f"train sims: {len(train_data)}, val sims: {len(val_data)}")

stats = compute_norm_stats(train_data)
save_norm_stats(stats, "checkpoints/norm_stats.json")
print("norm stats:", stats)

train_ds = SimPointCloudDataset(train_data, train_names, stats, n_points=32000, subsample=True)
loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_single)

t0 = time.time()
batch = next(iter(loader))
dt = time.time() - t0
x, y = batch["x"], batch["y"]
print(f"single batch load time: {dt:.4f}s, x shape {x.shape}, y shape {y.shape}")
# NOTE: per-sim inlet velocity (x cols 2,3) is constant within one simulation, so a
# single-sim batch trivially has std=0 on those two columns -- that's expected, not a bug.
# Normalization correctness has to be checked in aggregate across many sims instead.
assert dt < 1.0, "batch did not load in under 1s"

xs, ys = [], []
for i, b in enumerate(loader):
    xs.append(b["x"])
    ys.append(b["y"])
    if i >= 29:
        break
x_all = torch.cat(xs, dim=0)
y_all = torch.cat(ys, dim=0)
print(f"aggregate over {len(xs)} sims: x mean {x_all.mean(0).numpy()}, x std {x_all.std(0).numpy()}")
print(f"aggregate over {len(ys)} sims: y mean {y_all.mean(0).numpy()}, y std {y_all.std(0).numpy()}")
assert x_all.mean().abs() < 0.1 and abs(x_all.std().item() - 1.0) < 0.3, "x not normalized in aggregate"
assert y_all.mean().abs() < 0.1 and abs(y_all.std().item() - 1.0) < 0.3, "y not normalized in aggregate"

# sanity plot: pressure field of one training sim (raw units)
sim = train_data[0]
pos = sim[:, :2]
pressure = sim[:, 9]
fig, ax = plt.subplots(figsize=(7, 4))
sc = ax.scatter(pos[:, 0], pos[:, 1], c=pressure, s=1, cmap="viridis")
ax.set_aspect("equal")
ax.set_title(f"sanity check: pressure field, {train_names[0]}")
fig.colorbar(sc, ax=ax)
fig.tight_layout()
fig.savefig("plots/sanity_pressure_field.png", dpi=120)
print("saved plots/sanity_pressure_field.png")
print("GATE 3 CHECKS PASSED")
