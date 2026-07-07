"""Gate 4: side-by-side predicted vs true pressure field for one validation sim,
using the best (lowest val-loss) MLP checkpoint."""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, ".")
from src.dataset import (
    SimPointCloudDataset,
    load_norm_stats,
    load_scarce_cached,
    split_train_val,
    unnormalize_y,
)
from src.models import build_model

ckpt = torch.load("checkpoints/mlp_scarce_best.pt", map_location="cpu", weights_only=False)
print("checkpoint epoch:", ckpt["epoch"], "val_loss:", ckpt["val_loss"])

stats = load_norm_stats(ckpt["norm_stats_path"])
model = build_model(ckpt["model_name"], in_dim=7, out_dim=4)
model.load_state_dict(ckpt["model_state"])
model.eval()

data_list, name_list = load_scarce_cached(root="data/Dataset", train=True)
(_, _), (val_data, val_names) = split_train_val(data_list, name_list, n_val=20, seed=0)

sim = val_data[0]
name = val_names[0]
pos = sim[:, :2]
true_pressure = sim[:, 9]

x_raw = sim[:, :7]
x_norm = (x_raw - np.array(stats["x_mean"], dtype=np.float32)) / np.array(stats["x_std"], dtype=np.float32)
with torch.no_grad():
    pred_norm = model(torch.from_numpy(x_norm)).numpy()
pred_unnorm = unnormalize_y(pred_norm, stats)
pred_pressure = pred_unnorm[:, 2]

vmin, vmax = float(true_pressure.min()), float(true_pressure.max())
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True, sharey=True)
sc0 = axes[0].scatter(pos[:, 0], pos[:, 1], c=true_pressure, s=1.5, cmap="viridis", vmin=vmin, vmax=vmax)
axes[0].set_title(f"true pressure — {name}")
axes[0].set_aspect("equal")
sc1 = axes[1].scatter(pos[:, 0], pos[:, 1], c=pred_pressure, s=1.5, cmap="viridis", vmin=vmin, vmax=vmax)
axes[1].set_title(f"predicted pressure — epoch {ckpt['epoch']}")
axes[1].set_aspect("equal")
fig.colorbar(sc1, ax=axes, shrink=0.8)
fig.savefig("plots/gate4_pred_vs_true_pressure.png", dpi=130)
print("saved plots/gate4_pred_vs_true_pressure.png")

mse = float(np.mean((pred_pressure - true_pressure) ** 2))
print(f"pressure MSE (raw units) on this val sim: {mse:.2f}")
