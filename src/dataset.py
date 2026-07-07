"""AirfRANS loading, train-only normalization, float32 casting, per-epoch subsampling.

Column layout of the (N, 12) arrays returned by airfrans.dataset.load (verified against
the installed package source, site-packages/airfrans/dataset.py):
  [0:2]  position (x, y)
  [2:4]  inlet velocity (vx, vy)
  [4:5]  signed distance to airfoil
  [5:7]  normals (nx, ny), zero off-surface
  [7:9]  target velocity (vx, vy)
  [9:10] target pressure / rho
  [10:11] target turbulent kinematic viscosity
  [11:12] surface boolean
"""
import json

import numpy as np
import torch
from torch.utils.data import Dataset

INPUT_COLS = slice(0, 7)
TARGET_COLS = slice(7, 11)
SURFACE_COL = 11


def load_scarce(root="data/Dataset", train=True):
    import airfrans as af

    data_list, name_list = af.dataset.load(root=root, task="scarce", train=train)
    data_list = [d.astype(np.float32) for d in data_list]
    return data_list, name_list


def split_train_val(data_list, name_list, n_val=20, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data_list))
    val_idx = set(idx[:n_val].tolist())
    train_data, train_names, val_data, val_names = [], [], [], []
    for i, (d, n) in enumerate(zip(data_list, name_list)):
        if i in val_idx:
            val_data.append(d)
            val_names.append(n)
        else:
            train_data.append(d)
            train_names.append(n)
    return (train_data, train_names), (val_data, val_names)


def compute_norm_stats(train_data):
    all_pts = np.concatenate(train_data, axis=0)
    x = all_pts[:, INPUT_COLS]
    y = all_pts[:, TARGET_COLS]
    return {
        "x_mean": x.mean(axis=0).tolist(),
        "x_std": (x.std(axis=0) + 1e-8).tolist(),
        "y_mean": y.mean(axis=0).tolist(),
        "y_std": (y.std(axis=0) + 1e-8).tolist(),
    }


def save_norm_stats(stats, path):
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_norm_stats(path):
    with open(path) as f:
        return json.load(f)


def unnormalize_y(y_norm, stats):
    y_mean = np.asarray(stats["y_mean"], dtype=np.float32)
    y_std = np.asarray(stats["y_std"], dtype=np.float32)
    return y_norm * y_std + y_mean


class SimPointCloudDataset(Dataset):
    """One item = one simulation. Subsampled to n_points per __getitem__ call
    (train mode, different points every epoch via set_epoch); full resolution
    when subsample=False (validation/test)."""

    def __init__(self, data_list, names, stats, n_points=32000, subsample=True, seed=0):
        self.data_list = data_list
        self.names = names
        self.stats = stats
        self.n_points = n_points
        self.subsample = subsample
        self.epoch = seed
        self._x_mean = np.asarray(stats["x_mean"], dtype=np.float32)
        self._x_std = np.asarray(stats["x_std"], dtype=np.float32)
        self._y_mean = np.asarray(stats["y_mean"], dtype=np.float32)
        self._y_std = np.asarray(stats["y_std"], dtype=np.float32)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        arr = self.data_list[idx]
        n_total = arr.shape[0]
        if self.subsample and self.n_points < n_total:
            rng = np.random.default_rng(self.epoch * 100_003 + idx)
            sel = rng.choice(n_total, size=self.n_points, replace=False)
            arr = arr[sel]

        x_raw = arr[:, INPUT_COLS]
        y_raw = arr[:, TARGET_COLS]
        surface = arr[:, SURFACE_COL]

        x = (x_raw - self._x_mean) / self._x_std
        y = (y_raw - self._y_mean) / self._y_std

        return {
            "x": torch.from_numpy(x.astype(np.float32)),
            "y": torch.from_numpy(y.astype(np.float32)),
            "surface": torch.from_numpy(surface.astype(np.float32)),
            "position": torch.from_numpy(x_raw[:, :2].astype(np.float32)),
            "name": self.names[idx],
        }


def collate_single(batch):
    assert len(batch) == 1, "batch size must be 1 simulation per the build spec"
    return batch[0]
