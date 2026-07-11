"""Diagnostic: does the Cl regression on full-scale GraphSAGE come from
chunked_predict's cross-chunk k-NN discontinuity? Re-run eval with a chunk
size big enough that every test sim fits in one chunk (single k-NN graph
per sim, exactly matching how the model would be used in a real single-shot
prediction), and compare Cl/Cd against the chunk=50_000 result."""
import sys

sys.path.insert(0, ".")
import numpy as np
from scipy.stats import spearmanr

from src.dataset import INPUT_COLS, SURFACE_COL, TARGET_COLS, load_airfrans_cached
from src.evaluate import chunked_predict, load_model_from_checkpoint, predicted_force_coefficients

model, stats, ckpt = load_model_from_checkpoint("checkpoints/graphsage_full_64k_best.pt", device="cpu")
data_list, name_list = load_airfrans_cached(task="full", train=False)

cd_true_l, cd_pred_l, cl_true_l, cl_pred_l = [], [], [], []
for i, (arr, name) in enumerate(zip(data_list, name_list)):
    x_raw = arr[:, INPUT_COLS]
    y_pred = chunked_predict(model, x_raw, stats, "cpu", chunk=250_000)  # single chunk per sim
    (cd_pred, cl_pred), (cd_true, cl_true) = predicted_force_coefficients(
        "data/Dataset", name, y_pred[:, :2], y_pred[:, 2], y_pred[:, 3]
    )
    cd_true_l.append(cd_true)
    cd_pred_l.append(cd_pred)
    cl_true_l.append(cl_true)
    cl_pred_l.append(cl_pred)
    if i % 40 == 0 or i == len(data_list) - 1:
        print(f"[{i+1}/{len(data_list)}] {name}")

cd_true_a, cd_pred_a = np.array(cd_true_l), np.array(cd_pred_l)
cl_true_a, cl_pred_a = np.array(cl_true_l), np.array(cl_pred_l)
cl_rel_err = float(np.mean(np.abs(cl_pred_a - cl_true_a) / np.abs(cl_true_a)))
cl_spearman = float(spearmanr(cl_true_a, cl_pred_a).correlation)
cd_rel_err = float(np.mean(np.abs(cd_pred_a - cd_true_a) / np.abs(cd_true_a)))
cd_spearman = float(spearmanr(cd_true_a, cd_pred_a).correlation)

print(f"\nsingle-chunk (chunk=250000) results:")
print(f"cl_rel_err={cl_rel_err:.4f}  cl_spearman={cl_spearman:.4f}")
print(f"cd_rel_err={cd_rel_err:.4f}  cd_spearman={cd_spearman:.4f}")
print(f"\ncompare to chunk=50000 (original eval): cl_rel_err=1.2901 cl_spearman=0.8258 cd_rel_err=11.4284 cd_spearman=0.1331")
