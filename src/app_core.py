"""Non-UI prediction logic shared by app.py and its headless test (Gate 7).
Kept separate from app.py so it can be imported and exercised without
executing any Streamlit UI code."""
import numpy as np

from src.evaluate import chunked_predict, load_model_from_checkpoint
from src.geometry import apply_inflow, generate_geometry_cloud, integrate_forces, naca_airfoil, resample_close

CHECKPOINT = "checkpoints/graphsage_scarce_best.pt"


def parse_dat_bytes(data_bytes):
    text = data_bytes.decode("utf-8", errors="ignore")
    pts = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            x, y = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        pts.append((x, y))
    return np.array(pts, dtype=np.float64)


def load_geometry(airfoil_key, dat_bytes=None, n_volume=12000):
    if dat_bytes is not None:
        raw = parse_dat_bytes(dat_bytes)
    else:
        raw = naca_airfoil(airfoil_key, nb_samples=200)
    surface_coords, _chord = resample_close(raw, n_points=300)
    return generate_geometry_cloud(surface_coords, n_volume=n_volume, seed=0)


def predict_and_integrate(geo, reynolds, aoa_deg, model, stats):
    cloud = apply_inflow(geo, reynolds, aoa_deg)
    pred = chunked_predict(model, cloud["x"], stats, "cpu")

    is_surface = cloud["is_surface"]
    surface_pos = cloud["position"][is_surface]
    pred_surface = pred[is_surface]
    surface_normal_vecs = cloud["surface_normals"]

    eps = 0.01
    offset_pos = surface_pos - surface_normal_vecs * eps
    n_surf = surface_pos.shape[0]
    inlet_v = np.tile(cloud["x"][0, 2:4], (n_surf, 1))
    offset_x = np.concatenate(
        [offset_pos, inlet_v, np.full((n_surf, 1), eps, dtype=np.float32), np.zeros((n_surf, 2), dtype=np.float32)],
        axis=1,
    ).astype(np.float32)
    offset_pred = chunked_predict(model, offset_x, stats, "cpu")

    cd, cl = integrate_forces(
        surface_pos, surface_normal_vecs, pred_surface, offset_pred, eps, cloud["inlet_speed"], cloud["aoa_rad"]
    )
    return cloud, pred, cd, cl
