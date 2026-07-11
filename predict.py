"""CLI: predict flow field and force coefficients for an arbitrary airfoil.

Usage:
  python predict.py --naca 2412 --reynolds 4e6 --aoa 5.0
  python predict.py --dat foil.dat --reynolds 4e6 --aoa 5.0 --checkpoint checkpoints/mlp_scarce_best.pt
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluate import chunked_predict, load_model_from_checkpoint
from src.geometry import check_envelope, generate_point_cloud, integrate_forces, naca_airfoil, parse_dat_file, resample_close


def get_args():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dat", help="path to a Selig-format UIUC .dat file")
    src.add_argument("--naca", help="4 or 5-digit NACA code, e.g. 2412")
    p.add_argument("--reynolds", type=float, required=True)
    p.add_argument("--aoa", type=float, required=True, help="angle of attack in degrees")
    p.add_argument("--checkpoint", default="checkpoints/mlp_full_fullres_v4_best.pt")
    p.add_argument("--n-volume", type=int, default=20000)
    p.add_argument("--out-prefix", default=None, help="prefix for output plots; default derived from inputs")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = get_args()

    for w in check_envelope(args.reynolds, args.aoa):
        print(f"WARNING: {w}")
    if args.dat:
        print("WARNING: arbitrary .dat geometry — accuracy degrades outside the NACA 4/5-digit family the model trained on.")
        raw = parse_dat_file(args.dat)
        tag = os.path.splitext(os.path.basename(args.dat))[0]
    else:
        raw = naca_airfoil(args.naca, nb_samples=200)
        tag = f"naca{args.naca}"

    surface_coords, chord = resample_close(raw, n_points=400)
    cloud = generate_point_cloud(surface_coords, args.reynolds, args.aoa, n_volume=args.n_volume, seed=args.seed)

    model, stats, ckpt = load_model_from_checkpoint(args.checkpoint, device=args.device)
    pred = chunked_predict(model, cloud["x"], stats, args.device)

    is_surface = cloud["is_surface"]
    surface_pos = cloud["position"][is_surface]
    pred_surface = pred[is_surface]
    surface_normal_vecs = cloud["surface_normals"]

    eps = 0.01
    # surface_normal_vecs point inward (into the solid); step outward into the fluid.
    offset_pos = surface_pos - surface_normal_vecs * eps
    n_surf = surface_pos.shape[0]
    inlet_v = np.tile(cloud["x"][0, 2:4], (n_surf, 1))
    offset_x = np.concatenate(
        [offset_pos, inlet_v, np.full((n_surf, 1), eps, dtype=np.float32), np.zeros((n_surf, 2), dtype=np.float32)],
        axis=1,
    ).astype(np.float32)
    offset_pred = chunked_predict(model, offset_x, stats, args.device)

    cd, cl = integrate_forces(
        surface_pos, surface_normal_vecs, pred_surface, offset_pred, eps, cloud["inlet_speed"], cloud["aoa_rad"]
    )
    print(f"Predicted Cl = {cl:.4f}, Cd = {cd:.4f}  (Re={args.reynolds:.2e}, AoA={args.aoa} deg)")

    out_prefix = args.out_prefix or f"plots/predict_{tag}_re{args.reynolds:.0e}_aoa{args.aoa}"
    os.makedirs("plots", exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    sc = ax.scatter(cloud["position"][:, 0], cloud["position"][:, 1], c=pred[:, 2], s=2, cmap="viridis")
    ax.set_aspect("equal")
    ax.set_title(f"predicted pressure — {tag}, Re={args.reynolds:.2e}, AoA={args.aoa}deg\nCl={cl:.3f} Cd={cd:.4f}")
    fig.colorbar(sc, ax=ax)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_pressure.png", dpi=130)
    print(f"saved {out_prefix}_pressure.png")

    order = np.argsort(np.arctan2(surface_pos[:, 1] - 0.5, surface_pos[:, 0] - 0.5))
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(surface_pos[order, 0], -pred_surface[order, 2], ".", ms=3)
    ax2.set_xlabel("x/c")
    ax2.set_ylabel("-Cp (arbitrary units, pressure)")
    ax2.set_title(f"predicted surface pressure — {tag}")
    fig2.tight_layout()
    fig2.savefig(f"{out_prefix}_cp.png", dpi=130)
    print(f"saved {out_prefix}_cp.png")


if __name__ == "__main__":
    main()
