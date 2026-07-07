"""Gate 5 dashboards: predicted-vs-true Cl/Cd scatter, and error contour plots
for the best/worst test airfoils by pressure MSE."""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from src.dataset import INPUT_COLS, TARGET_COLS, load_scarce_cached
from src.evaluate import chunked_predict, load_model_from_checkpoint

results = json.load(open("checkpoints/eval_results_mlp.json"))
per_sim = results["per_sim"]

cd_true = np.array([r["cd_true"] for r in per_sim])
cd_pred = np.array([r["cd_pred"] for r in per_sim])
cl_true = np.array([r["cl_true"] for r in per_sim])
cl_pred = np.array([r["cl_pred"] for r in per_sim])

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].scatter(cl_true, cl_pred, s=12, alpha=0.6)
lims = [min(cl_true.min(), cl_pred.min()), max(cl_true.max(), cl_pred.max())]
axes[0].plot(lims, lims, "r--", lw=1)
axes[0].set_xlabel("true Cl")
axes[0].set_ylabel("predicted Cl")
axes[0].set_title(f"Cl (Spearman={results['cl_spearman']:.3f})")

axes[1].scatter(cd_true, cd_pred, s=12, alpha=0.6, color="darkorange")
lims = [min(cd_true.min(), cd_pred.min()), max(cd_true.max(), cd_pred.max())]
axes[1].plot(lims, lims, "r--", lw=1)
axes[1].set_xlabel("true Cd")
axes[1].set_ylabel("predicted Cd")
axes[1].set_title(f"Cd (Spearman={results['cd_spearman']:.3f})")
fig.tight_layout()
fig.savefig("plots/gate5_cl_cd_scatter.png", dpi=130)
print("saved plots/gate5_cl_cd_scatter.png")

# error contour: best and worst test sim by pressure MSE
pressure_mse = np.array([r["pressure_mse"] for r in per_sim])
worst_idx = int(np.argmax(pressure_mse))
best_idx = int(np.argmin(pressure_mse))
worst_name = per_sim[worst_idx]["name"]
best_name = per_sim[best_idx]["name"]
print(f"best sim: {best_name} (pressure_mse={pressure_mse[best_idx]:.1f})")
print(f"worst sim: {worst_name} (pressure_mse={pressure_mse[worst_idx]:.1f})")

model, stats, ckpt = load_model_from_checkpoint("checkpoints/mlp_scarce_best.pt", device="cpu")
data_list, name_list = load_scarce_cached(root="data/Dataset", train=False)
name_to_arr = dict(zip(name_list, data_list))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, name, label in zip(axes, [best_name, worst_name], ["best", "worst"]):
    arr = name_to_arr[name]
    x_raw = arr[:, INPUT_COLS]
    y_true = arr[:, TARGET_COLS]
    y_pred = chunked_predict(model, x_raw, stats, "cpu")
    err = np.abs(y_pred[:, 2] - y_true[:, 2])
    pos = arr[:, :2]
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=err, s=2, cmap="inferno")
    ax.set_title(f"{label} test sim — |pressure error|\n{name}")
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig("plots/gate5_error_contours.png", dpi=130)
print("saved plots/gate5_error_contours.png")
