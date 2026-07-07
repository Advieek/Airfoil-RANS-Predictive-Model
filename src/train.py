"""Training loop for the airfoil surrogate: TensorBoard logging + evolution
snapshots from the start (per BUILD_SPEC Step 3/5). Supports a synthetic
smoke-test mode (Gate 2, no real data needed) and the real AirfRANS pipeline
(Gate 3+).

Usage:
  python -m src.train --mode synthetic --epochs 3
  python -m src.train --mode real --model mlp --epochs 400 --run-name mlp_scarce
"""
import argparse
import csv
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.models import build_model


def get_device(name="mps"):
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_synthetic_data(n_sims=4, n_points=2000, seed=0):
    """Deterministic linear map + noise so loss has something real to chase,
    just enough to prove the training/logging mechanics on MPS."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(7, 4)).astype(np.float32)
    sims = []
    for i in range(n_sims):
        pos = rng.uniform(-1, 1, size=(n_points, 2)).astype(np.float32)
        rest = rng.normal(size=(n_points, 5)).astype(np.float32)
        x = np.concatenate([pos, rest], axis=1).astype(np.float32)
        y = x @ A + 0.01 * rng.normal(size=(n_points, 4)).astype(np.float32)
        sims.append(
            {
                "x": torch.from_numpy(x),
                "y": torch.from_numpy(y),
                "position": torch.from_numpy(pos),
                "name": f"synthetic_{i}",
            }
        )
    return sims


def save_evolution_frame(position, values, vmin, vmax, epoch, out_dir, tag="pressure"):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(position[:, 0], position[:, 1], c=values, s=2, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(f"{tag} — epoch {epoch}")
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax)
    fig.tight_layout()
    path = os.path.join(out_dir, f"epoch_{epoch:04d}.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def run_synthetic(args):
    device = get_device(args.device)
    sims = make_synthetic_data()
    model = build_model(args.model, in_dim=7, out_dim=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_dir = os.path.join("runs", args.run_name)
    writer = SummaryWriter(run_dir)

    evo_sim = sims[0]
    evo_pressure = evo_sim["y"][:, 2].numpy()
    vmin, vmax = float(evo_pressure.min()), float(evo_pressure.max())

    losses = []
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for sim in sims:
            x = sim["x"].to(device)
            y = sim["y"].to(device)
            edge_index = model.build_graph(x[:, :2]) if hasattr(model, "build_graph") else None
            pred = model(x, edge_index) if edge_index is not None else model(x)
            loss = torch.nn.functional.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        epoch_loss /= len(sims)
        losses.append(epoch_loss)
        writer.add_scalar("loss/train_synthetic", epoch_loss, epoch)

        with torch.no_grad():
            x = evo_sim["x"].to(device)
            edge_index = model.build_graph(x[:, :2]) if hasattr(model, "build_graph") else None
            pred = model(x, edge_index) if edge_index is not None else model(x)
            pred_pressure = pred[:, 2].cpu().numpy()
        save_evolution_frame(evo_sim["position"].numpy(), pred_pressure, vmin, vmax, epoch, "plots/evolution")

    writer.close()
    print(f"synthetic losses: {losses}")
    assert losses[-1] < losses[0], "loss did not decrease on synthetic smoke test"
    return losses


def run_real(args):
    from src.dataset import (
        SimPointCloudDataset,
        collate_single,
        compute_norm_stats,
        load_scarce,
        save_norm_stats,
        split_train_val,
        unnormalize_y,
    )

    device = get_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t0 = time.time()
    data_list, name_list = load_scarce(root=args.data_root, train=True)
    print(f"loaded {len(data_list)} sims in {time.time() - t0:.1f}s; shape[0]={data_list[0].shape}")

    (train_data, train_names), (val_data, val_names) = split_train_val(
        data_list, name_list, n_val=args.n_val, seed=args.seed
    )
    print(f"train sims: {len(train_data)}, val sims: {len(val_data)}")

    stats = compute_norm_stats(train_data)
    os.makedirs("checkpoints", exist_ok=True)
    save_norm_stats(stats, "checkpoints/norm_stats.json")

    train_ds = SimPointCloudDataset(train_data, train_names, stats, n_points=args.n_points, subsample=True)
    val_ds = SimPointCloudDataset(val_data, val_names, stats, n_points=args.n_points, subsample=True)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_single)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_single)

    model = build_model(args.model, in_dim=7, out_dim=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    run_dir = os.path.join("runs", args.run_name)
    writer = SummaryWriter(run_dir)
    csv_path = os.path.join(run_dir, "losses.csv")
    os.makedirs(run_dir, exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "val_loss"])

    evo_idx = 0
    evo_name = val_names[evo_idx]
    evo_arr = val_data[evo_idx]
    evo_position = evo_arr[:, :2]
    evo_pressure_true = evo_arr[:, 9]
    vmin, vmax = float(evo_pressure_true.min()), float(evo_pressure_true.max())
    evo_x_norm = (evo_arr[:, :7] - np.array(stats["x_mean"], dtype=np.float32)) / np.array(
        stats["x_std"], dtype=np.float32
    )

    best_val = float("inf")
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"{args.run_name}_best.pt")

    for epoch in range(args.epochs):
        train_ds.set_epoch(epoch)
        model.train()
        train_loss_sum, n_sims = 0.0, 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            edge_index = model.build_graph(x[:, :2]) if hasattr(model, "build_graph") else None
            pred = model(x, edge_index) if edge_index is not None else model(x)
            loss = torch.nn.functional.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss_sum += loss.item()
            n_sims += 1
        sched.step()
        train_loss = train_loss_sum / n_sims

        model.eval()
        val_loss_sum, n_val_sims = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                edge_index = model.build_graph(x[:, :2]) if hasattr(model, "build_graph") else None
                pred = model(x, edge_index) if edge_index is not None else model(x)
                loss = torch.nn.functional.mse_loss(pred, y)
                val_loss_sum += loss.item()
                n_val_sims += 1
        val_loss = val_loss_sum / n_val_sims

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        if epoch % 20 == 0:
            for name, p in model.named_parameters():
                writer.add_histogram(f"weights/{name}", p, epoch)
        csv_writer.writerow([epoch, train_loss, val_loss])
        csv_file.flush()

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": args.model,
                    "norm_stats_path": "checkpoints/norm_stats.json",
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                ckpt_path,
            )

        if epoch % args.evolution_every == 0:
            model.eval()
            with torch.no_grad():
                x = torch.from_numpy(evo_x_norm).to(device)
                edge_index = model.build_graph(x[:, :2]) if hasattr(model, "build_graph") else None
                pred = model(x, edge_index) if edge_index is not None else model(x)
                pred_np = pred.cpu().numpy()
            pred_unnorm = unnormalize_y(pred_np, stats)
            pred_pressure = pred_unnorm[:, 2]
            save_evolution_frame(
                evo_position, pred_pressure, vmin, vmax, epoch, f"plots/evolution/{args.run_name}"
            )

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:4d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    csv_file.close()
    writer.close()
    print(f"best val loss: {best_val:.4f}; checkpoint: {ckpt_path}; evolution sim: {evo_name}")
    return best_val, ckpt_path


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["synthetic", "real"], default="real")
    p.add_argument("--model", choices=["mlp", "graphsage"], default="mlp")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-points", type=int, default=32000)
    p.add_argument("--n-val", type=int, default=20)
    p.add_argument("--data-root", default="data/Dataset")
    p.add_argument("--run-name", default="mlp_scarce")
    p.add_argument("--evolution-every", type=int, default=10)
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.mode == "synthetic":
        run_synthetic(args)
    else:
        run_real(args)
