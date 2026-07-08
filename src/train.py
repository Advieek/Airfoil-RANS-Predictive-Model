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
import gc
import json
import os
import resource
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


def collate_list(batch):
    """Multi-sim batching: keep each sim's variable-length point cloud
    separate (no padding/stacking) -- the training loop concatenates them
    (and, for GraphSAGE, builds a block-diagonal k-NN graph) after the .to(device)
    transfer."""
    return batch


def assemble_batch(model, batch_list, device):
    """Concatenate a multi-sim batch into one big point set for a single
    forward/backward pass. For GraphSAGE this builds one block-diagonal graph
    (each sim's own k-NN graph, no cross-sim edges) via build_batched_graph."""
    xs = [b["x"].to(device) for b in batch_list]
    ys = [b["y"].to(device) for b in batch_list]
    x_cat = torch.cat(xs, dim=0)
    y_cat = torch.cat(ys, dim=0)
    if hasattr(model, "build_graph"):
        from src.models import build_batched_graph

        edge_index = build_batched_graph(model, [x[:, :2] for x in xs])
        pred = model(x_cat, edge_index)
    else:
        pred = model(x_cat)
    return pred, y_cat


def run_real(args):
    from src.dataset import (
        SimPointCloudDataset,
        collate_single,
        compute_norm_stats,
        load_airfrans_cached,
        save_norm_stats,
        split_train_val,
        unnormalize_y,
    )

    device = get_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t0 = time.time()
    data_list, name_list = load_airfrans_cached(root=args.data_root, task=args.task, train=True)
    print(f"loaded {len(data_list)} '{args.task}' sims in {time.time() - t0:.1f}s; shape[0]={data_list[0].shape}")

    n_val = args.n_val if args.n_val is not None else max(1, round(0.1 * len(data_list)))
    (train_data, train_names), (val_data, val_names) = split_train_val(data_list, name_list, n_val=n_val, seed=args.seed)
    print(f"train sims: {len(train_data)}, val sims: {len(val_data)}")

    stats = compute_norm_stats(train_data)
    os.makedirs("checkpoints", exist_ok=True)
    save_norm_stats(stats, "checkpoints/norm_stats.json")

    subsample = not args.full_resolution
    n_points = args.n_points
    train_ds = SimPointCloudDataset(train_data, train_names, stats, n_points=n_points, subsample=subsample)
    val_ds = SimPointCloudDataset(val_data, val_names, stats, n_points=n_points, subsample=subsample)

    sims_per_batch = args.sims_per_batch
    train_loader = DataLoader(train_ds, batch_size=sims_per_batch, shuffle=True, collate_fn=collate_list)
    val_loader = DataLoader(val_ds, batch_size=sims_per_batch, shuffle=False, collate_fn=collate_list)

    if args.model == "mlp":
        hidden = tuple(int(h) for h in args.hidden.split(","))
        model_kwargs = {"in_dim": 7, "out_dim": 4, "hidden": hidden}
    else:
        model_kwargs = {
            "in_dim": 7,
            "out_dim": 4,
            "hidden": args.gnn_hidden,
            "n_layers": args.gnn_layers,
            "k": args.gnn_k,
            "knn_backend": args.gnn_knn_backend,
        }
    model = build_model(args.model, **model_kwargs).to(device)
    if args.init_from:
        init_ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(init_ckpt["model_state"])
        print(f"warm-started from {args.init_from} (epoch {init_ckpt['epoch']}, val_loss {init_ckpt['val_loss']:.4f})")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={args.model} params={n_params:,} kwargs={model_kwargs}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"{args.run_name}_best.pt")
    resume_ckpt_path = os.path.join(args.out_dir, f"{args.run_name}_resume.pt")

    start_epoch = 0
    best_val = float("inf")
    if args.resume_from:
        # Full training-state restore (model + optimizer + scheduler + epoch counter),
        # unlike --init-from which only warm-starts the weights into a fresh 400-epoch
        # schedule. This is what makes a crash genuinely resumable instead of costing
        # most of a day's progress each time: see PROGRESS.md 2026-07-08 entries.
        resume_ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_ckpt["model_state"])
        opt.load_state_dict(resume_ckpt["optimizer_state"])
        sched.load_state_dict(resume_ckpt["scheduler_state"])
        start_epoch = resume_ckpt["epoch"] + 1
        best_val = resume_ckpt["best_val"]
        print(
            f"resumed from {args.resume_from}: continuing at epoch {start_epoch}/{args.epochs}, "
            f"best_val so far {best_val:.4f}"
        )

    run_dir = os.path.join("runs", args.run_name)
    writer = SummaryWriter(run_dir)
    csv_path = os.path.join(run_dir, "losses.csv")
    os.makedirs(run_dir, exist_ok=True)
    csv_file = open(csv_path, "a" if args.resume_from else "w", newline="")
    csv_writer = csv.writer(csv_file)
    if not args.resume_from:
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

    def save_resume_checkpoint(epoch):
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "scheduler_state": sched.state_dict(),
                "model_name": args.model,
                "model_kwargs": model_kwargs,
                "norm_stats_path": "checkpoints/norm_stats.json",
                "epoch": epoch,
                "best_val": best_val,
            },
            resume_ckpt_path,
        )

    for epoch in range(start_epoch, args.epochs):
        train_ds.set_epoch(epoch)
        model.train()
        train_loss_sum, n_steps = 0.0, 0
        for batch_list in train_loader:
            pred, y_cat = assemble_batch(model, batch_list, device)
            loss = torch.nn.functional.mse_loss(pred, y_cat)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss_sum += loss.item()
            n_steps += 1
            del pred, y_cat, loss
            if device.type == "mps" and n_steps % 5 == 0:
                # MPS's caching allocator retains freed blocks per unique shape for reuse.
                # Full-resolution batches concatenate a different combination of
                # variable-length sims every step, so shapes rarely repeat -- without
                # periodic release the cache grows effectively unbounded over a multi-hour
                # run (observed: one process reached 219GB RSS / 139GB compressed on this
                # 64GB machine, near-zero free memory, before this fix; a first attempt at
                # a fix with a 20-step interval and no gc.collect() slowed but did not
                # eliminate the growth -- it still silently died, likely OOM-killed by the
                # kernel, around epoch 235). Pairing with gc.collect() covers the case where
                # a reference cycle (e.g. in the autograd graph) delays Python from dropping
                # a tensor's refcount to zero before the allocator can reclaim it.
                gc.collect()
                torch.mps.empty_cache()
        sched.step()
        train_loss = train_loss_sum / n_steps
        if device.type == "mps":
            gc.collect()
            torch.mps.empty_cache()

        model.eval()
        val_loss_sum, n_val_steps = 0.0, 0
        with torch.no_grad():
            for batch_list in val_loader:
                pred, y_cat = assemble_batch(model, batch_list, device)
                loss = torch.nn.functional.mse_loss(pred, y_cat)
                val_loss_sum += loss.item()
                n_val_steps += 1
                del pred, y_cat, loss
                if device.type == "mps" and n_val_steps % 5 == 0:
                    gc.collect()
                    torch.mps.empty_cache()
        val_loss = val_loss_sum / n_val_steps
        if device.type == "mps":
            gc.collect()
            torch.mps.empty_cache()

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        if epoch % 20 == 0:
            for name, p in model.named_parameters():
                writer.add_histogram(f"weights/{name}", p, epoch)
        csv_writer.writerow([epoch, train_loss, val_loss])
        csv_file.flush()

        rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
        writer.add_scalar("system/rss_gb", rss_gb, epoch)
        if rss_gb > args.rss_limit_gb:
            print(
                f"epoch {epoch}: RSS {rss_gb:.1f}GB exceeded --rss-limit-gb {args.rss_limit_gb} "
                "-- stopping cleanly to avoid a silent OOM-kill. Saving a full resume checkpoint; "
                f"continue with --resume-from {resume_ckpt_path}."
            )
            save_resume_checkpoint(epoch)
            csv_file.close()
            writer.close()
            return best_val, ckpt_path

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": args.model,
                    "model_kwargs": model_kwargs,
                    "norm_stats_path": "checkpoints/norm_stats.json",
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                ckpt_path,
            )

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs - 1:
            save_resume_checkpoint(epoch)

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
    p.add_argument("--task", choices=["scarce", "full"], default="full")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-points", type=int, default=32000, help="ignored if --full-resolution is set")
    p.add_argument("--full-resolution", action="store_true", help="use every point per sim, no subsampling")
    p.add_argument("--sims-per-batch", type=int, default=4)
    p.add_argument("--hidden", default="256,256,256,256,256", help="comma-separated MLP hidden layer widths")
    p.add_argument("--gnn-hidden", type=int, default=256)
    p.add_argument("--gnn-layers", type=int, default=6)
    p.add_argument("--gnn-k", type=int, default=16)
    p.add_argument("--gnn-knn-backend", choices=["mps_chunked", "cpu_kdtree"], default="mps_chunked")
    p.add_argument("--n-val", type=int, default=None, help="default: 10%% of train sims")
    p.add_argument("--init-from", default=None, help="warm-start model weights only (fresh optimizer/schedule/epoch 0)")
    p.add_argument(
        "--resume-from",
        default=None,
        help="fully resume training (model+optimizer+scheduler+epoch counter) from a "
        "--checkpoint-every / RSS-limit resume checkpoint; continues the same run's CSV/TensorBoard log",
    )
    p.add_argument("--checkpoint-every", type=int, default=10, help="save a full resumable checkpoint every N epochs")
    p.add_argument(
        "--rss-limit-gb",
        type=float,
        default=40.0,
        help="stop cleanly (best checkpoint already saved) if peak RSS exceeds this, rather than risk a silent kernel OOM-kill",
    )
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
