"""Re-run the scarce-task GraphSAGE evaluation with the chunking-artifact fix
(src/evaluate.py, 2026-07-11) and force knn_backend='cpu_kdtree' explicitly,
since this checkpoint predates the knn_backend field in model_kwargs and
would otherwise default to the much slower mps_chunked path (repeated
GPU-attempt-then-fallback per sim -- the same pathological slowness
diagnosed during training)."""
import json
import sys

sys.path.insert(0, ".")
import numpy as np
from scipy.stats import spearmanr

from src.dataset import INPUT_COLS, SURFACE_COL, TARGET_COLS, load_airfrans_cached
from src.evaluate import chunked_predict, load_model_from_checkpoint, predicted_force_coefficients

model, stats, ckpt = load_model_from_checkpoint("checkpoints/graphsage_scarce_best.pt", device="cpu")
model.knn_backend = "cpu_kdtree"
data_list, name_list = load_airfrans_cached(task="full", train=False)

field_se_volume, field_se_surface = [], []
cd_true_l, cd_pred_l, cl_true_l, cl_pred_l = [], [], [], []
per_sim_records = []

for i, (arr, name) in enumerate(zip(data_list, name_list)):
    x_raw = arr[:, INPUT_COLS]
    y_true = arr[:, TARGET_COLS]
    surface = arr[:, SURFACE_COL].astype(bool)

    y_pred = chunked_predict(model, x_raw, stats, "cpu")

    se = (y_pred - y_true) ** 2
    field_se_volume.append(se.mean(axis=0))
    if surface.sum() > 0:
        field_se_surface.append(se[surface].mean(axis=0))

    (cd_pred, cl_pred), (cd_true, cl_true) = predicted_force_coefficients(
        "data/Dataset", name, y_pred[:, :2], y_pred[:, 2], y_pred[:, 3]
    )
    cd_true_l.append(cd_true)
    cd_pred_l.append(cd_pred)
    cl_true_l.append(cl_true)
    cl_pred_l.append(cl_pred)
    per_sim_records.append(
        {"name": name, "cd_true": cd_true, "cd_pred": cd_pred, "cl_true": cl_true, "cl_pred": cl_pred, "pressure_mse": float(se[:, 2].mean())}
    )
    if i % 40 == 0 or i == len(data_list) - 1:
        print(f"[{i+1}/{len(data_list)}] {name}")

field_mse_volume = np.mean(field_se_volume, axis=0)
field_mse_surface = np.mean(field_se_surface, axis=0)
cd_true_a, cd_pred_a = np.array(cd_true_l), np.array(cd_pred_l)
cl_true_a, cl_pred_a = np.array(cl_true_l), np.array(cl_pred_l)

results = {
    "n_test_sims": len(data_list),
    "field_mse_volume": field_mse_volume.tolist(),
    "field_mse_surface": field_mse_surface.tolist(),
    "cd_rel_err": float(np.mean(np.abs(cd_pred_a - cd_true_a) / np.abs(cd_true_a))),
    "cl_rel_err": float(np.mean(np.abs(cl_pred_a - cl_true_a) / np.abs(cl_true_a))),
    "cd_spearman": float(spearmanr(cd_true_a, cd_pred_a).correlation),
    "cl_spearman": float(spearmanr(cl_true_a, cl_pred_a).correlation),
    "per_sim": per_sim_records,
}
with open("checkpoints/eval_results_graphsage.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps({k: v for k, v in results.items() if k != "per_sim"}, indent=2))
