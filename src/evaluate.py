"""Chunked full-resolution inference + force-coefficient evaluation on the
withheld AirfRANS test set.

Force coefficients are computed by reusing airfrans.Simulation's own
integration (wallshearstress -> force -> force_coefficient): we instantiate a
Simulation for the test sim's name (this reads the ground-truth VTU/VTP and
computes position/sdf/normals in the same node order as
airfrans.dataset.load), then monkey-patch its .velocity/.pressure/.nu_t
attributes with our model's predictions before calling
force_coefficient(reference=False). force_coefficient(reference=True) reads
straight from the VTU/VTP untouched, giving the ground-truth coefficients from
the same Simulation instance so node ordering can never mismatch between the
two calls.

Caveat: for GraphSAGE, chunked inference builds a separate k-NN graph per
chunk, so cross-chunk neighbor edges are missing at chunk boundaries. Noted,
not fixed today (full-res eval is dominated by MLP results in Step 6).
"""
import numpy as np
import torch
from scipy.stats import spearmanr

from src.dataset import INPUT_COLS, SURFACE_COL, TARGET_COLS, load_norm_stats, load_scarce_cached, unnormalize_y
from src.models import build_model


def chunked_predict(model, x_raw, stats, device, chunk=50_000):
    x_mean = np.asarray(stats["x_mean"], dtype=np.float32)
    x_std = np.asarray(stats["x_std"], dtype=np.float32)
    x_norm = (x_raw - x_mean) / x_std
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, x_norm.shape[0], chunk):
            xb = torch.from_numpy(x_norm[i : i + chunk]).to(device)
            if hasattr(model, "build_graph"):
                edge_index = model.build_graph(xb[:, :2])
                pb = model(xb, edge_index)
            else:
                pb = model(xb)
            preds.append(pb.cpu().numpy())
    pred_norm = np.concatenate(preds, axis=0)
    return unnormalize_y(pred_norm, stats)


def load_model_from_checkpoint(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model_name"], in_dim=7, out_dim=4).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    stats = load_norm_stats(ckpt["norm_stats_path"])
    return model, stats, ckpt


def predicted_force_coefficients(root, name, pred_velocity, pred_pressure, pred_nu_t):
    import airfrans as af

    sim = af.Simulation(root=root, name=name)
    sim.velocity = pred_velocity.astype(np.float64)
    sim.pressure = pred_pressure.reshape(-1, 1).astype(np.float64)
    sim.nu_t = pred_nu_t.reshape(-1, 1).astype(np.float64)
    (cd_pred, _, _), (cl_pred, _, _) = sim.force_coefficient(reference=False)
    (cd_true, _, _), (cl_true, _, _) = sim.force_coefficient(reference=True)
    return (float(cd_pred), float(cl_pred)), (float(cd_true), float(cl_true))


def evaluate_test_set(ckpt_path, data_root="data/Dataset", device="cpu", n_sims=None, verbose=True):
    model, stats, ckpt = load_model_from_checkpoint(ckpt_path, device)
    data_list, name_list = load_scarce_cached(root=data_root, train=False)
    if n_sims is not None:
        data_list, name_list = data_list[:n_sims], name_list[:n_sims]

    field_se_volume, field_se_surface = [], []
    cd_true_l, cd_pred_l, cl_true_l, cl_pred_l = [], [], [], []
    per_sim_records = []

    for i, (arr, name) in enumerate(zip(data_list, name_list)):
        x_raw = arr[:, INPUT_COLS]
        y_true = arr[:, TARGET_COLS]
        surface = arr[:, SURFACE_COL].astype(bool)

        y_pred = chunked_predict(model, x_raw, stats, device)

        se = (y_pred - y_true) ** 2
        field_se_volume.append(se.mean(axis=0))
        if surface.sum() > 0:
            field_se_surface.append(se[surface].mean(axis=0))

        (cd_pred, cl_pred), (cd_true, cl_true) = predicted_force_coefficients(
            data_root, name, y_pred[:, :2], y_pred[:, 2], y_pred[:, 3]
        )
        cd_true_l.append(cd_true)
        cd_pred_l.append(cd_pred)
        cl_true_l.append(cl_true)
        cl_pred_l.append(cl_pred)
        per_sim_records.append(
            {
                "name": name,
                "cd_true": cd_true,
                "cd_pred": cd_pred,
                "cl_true": cl_true,
                "cl_pred": cl_pred,
                "pressure_mse": float(se[:, 2].mean()),
            }
        )
        if verbose and (i % 20 == 0 or i == len(data_list) - 1):
            print(f"[{i+1}/{len(data_list)}] {name}: cd_true={cd_true:.4f} cd_pred={cd_pred:.4f}")

    field_mse_volume = np.mean(field_se_volume, axis=0)
    field_mse_surface = np.mean(field_se_surface, axis=0) if field_se_surface else None

    cd_true_a, cd_pred_a = np.array(cd_true_l), np.array(cd_pred_l)
    cl_true_a, cl_pred_a = np.array(cl_true_l), np.array(cl_pred_l)

    cd_rel_err = float(np.mean(np.abs(cd_pred_a - cd_true_a) / np.abs(cd_true_a)))
    cl_rel_err = float(np.mean(np.abs(cl_pred_a - cl_true_a) / np.abs(cl_true_a)))
    cd_spearman = float(spearmanr(cd_true_a, cd_pred_a).correlation)
    cl_spearman = float(spearmanr(cl_true_a, cl_pred_a).correlation)

    return {
        "n_test_sims": len(data_list),
        "field_mse_volume": field_mse_volume.tolist(),
        "field_mse_surface": field_mse_surface.tolist() if field_mse_surface is not None else None,
        "cd_rel_err": cd_rel_err,
        "cl_rel_err": cl_rel_err,
        "cd_spearman": cd_spearman,
        "cl_spearman": cl_spearman,
        "per_sim": per_sim_records,
    }


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/mlp_scarce_best.pt")
    p.add_argument("--n-sims", type=int, default=None)
    p.add_argument("--out", default="checkpoints/eval_results.json")
    args = p.parse_args()

    results = evaluate_test_set(args.checkpoint, n_sims=args.n_sims)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: v for k, v in results.items() if k != "per_sim"}, indent=2))
