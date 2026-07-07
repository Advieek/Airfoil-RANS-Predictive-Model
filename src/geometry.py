"""Arbitrary-airfoil point cloud generation and force integration for inference
on shapes the model never saw during training (Step 7).

Unlike Step 6 (real AirfRANS test sims, which have an OpenFOAM mesh so we can
reuse airfrans.Simulation's own wall-shear/force integration), arbitrary
airfoils here have no mesh -- just a parametrized surface curve. So force
integration is a simplified panel method: pressure integrated exactly around
the loop, and wall shear approximated from the model's own predicted
tangential-velocity gradient a small distance off the surface (since we can't
compute a full velocity-gradient tensor without mesh connectivity). This is
deliberately simpler than Step 6's evaluation and is documented as such.
"""
import numpy as np
import shapely
from scipy.interpolate import splev, splprep
from shapely.geometry import Polygon

AIR_T = 298.15  # Kelvin, matches airfrans.Simulation default
P_REF = 1.01325e5
MOL = 28.965338e-3

# Training envelope (per BUILD_SPEC / AirfRANS paper)
ENVELOPE_RE = (2e6, 6e6)
ENVELOPE_AOA = (-5.0, 15.0)


def air_kinematic_viscosity(T=AIR_T):
    return -3.400747e-6 + 3.452139e-8 * T + 1.00881778e-10 * T**2 - 1.363528e-14 * T**3


def air_density(T=AIR_T):
    return P_REF * MOL / (8.3144621 * T)


def parse_dat_file(path):
    """Parse a Selig-format UIUC .dat file: optional name header line, then x y pairs."""
    pts = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            pts.append((x, y))
    return np.array(pts, dtype=np.float64)


def naca_airfoil(code, nb_samples=200):
    """4 or 5-digit NACA code string, e.g. '2412'."""
    import airfrans.naca_generator as ng

    code = code.strip()
    if len(code) == 4:
        params = (int(code[0]), int(code[1]), int(code[2:]))
    elif len(code) == 5:
        params = (int(code[0]), int(code[1]), int(code[2]), int(code[3:]))
    else:
        raise ValueError("NACA code must be 4 or 5 digits")
    return ng.naca_generator(params, nb_samples=nb_samples, verbose=False)


def resample_close(coords, n_points=400):
    """Spline-resample a closed airfoil loop to n_points, chord-normalize x to
    [0, 1] (chord = x_max - x_min), close the trailing edge."""
    coords = np.asarray(coords, dtype=np.float64)
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    chord = x_max - x_min
    coords = coords.copy()
    coords[:, 0] = (coords[:, 0] - x_min) / chord
    coords[:, 1] = coords[:, 1] / chord

    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])

    tck, _ = splprep([coords[:, 0], coords[:, 1]], s=0, per=True)
    u_new = np.linspace(0, 1, n_points, endpoint=False)
    x_new, y_new = splev(u_new, tck)
    out = np.stack([x_new, y_new], axis=-1)
    out = np.vstack([out, out[0]])
    return out, chord


def surface_normals(coords):
    """Unit normals for a closed loop (coords[-1] == coords[0]), oriented INWARD
    (pointing from the surface into the airfoil solid) to match the AirfRANS
    training data convention -- verified against real training sims: e.g. the
    topmost point (clearly upper surface) has normal ~(0, -1), pointing down
    into the body, not up into the fluid. Getting this backwards silently
    feeds the model out-of-distribution normals and corrupts every prediction
    that touches the surface."""
    pts = coords[:-1]
    tangents = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    normals = np.stack([tangents[:, 1], -tangents[:, 0]], axis=-1)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.clip(norm, 1e-12, None)
    centroid = pts.mean(axis=0)
    sign = -np.sign(np.sum(normals * (pts - centroid), axis=1))
    normals = normals * sign[:, None]
    return np.vstack([normals, normals[0]])


def generate_point_cloud(
    surface_coords, reynolds, aoa_deg, n_volume=20000, domain=((-2, 4), (-1.5, 1.5)), seed=0, T=AIR_T
):
    """surface_coords: (M+1, 2) closed loop, chord-normalized to [0, 1] in x.
    Returns per-point input features [pos(2), inlet_v(2), sdf(1), normals(2)]
    matching the training column layout, plus bookkeeping fields."""
    rng = np.random.default_rng(seed)
    poly = Polygon(surface_coords)
    boundary = poly.exterior

    nu = air_kinematic_viscosity(T)
    inlet_speed = reynolds * nu  # chord = 1
    aoa_rad = np.deg2rad(aoa_deg)
    inlet_vx, inlet_vy = inlet_speed * np.cos(aoa_rad), inlet_speed * np.sin(aoa_rad)

    surf_pts = surface_coords[:-1]
    surf_norm = surface_normals(surface_coords)[:-1]
    n_surf = len(surf_pts)

    n_near = int(n_volume * 0.6)
    idx = rng.integers(0, n_surf, size=n_near)
    offsets = rng.exponential(scale=0.03, size=n_near)
    # surf_norm points inward (into the solid); step outward into the fluid instead.
    near_pts = surf_pts[idx] - surf_norm[idx] * offsets[:, None]

    n_far = n_volume - n_near
    far_pts = np.stack(
        [rng.uniform(domain[0][0], domain[0][1], size=n_far), rng.uniform(domain[1][0], domain[1][1], size=n_far)],
        axis=-1,
    )

    cand = np.vstack([near_pts, far_pts])
    mask_domain = (
        (cand[:, 0] >= domain[0][0])
        & (cand[:, 0] <= domain[0][1])
        & (cand[:, 1] >= domain[1][0])
        & (cand[:, 1] <= domain[1][1])
    )
    cand = cand[mask_domain]
    inside = shapely.contains_xy(poly, cand[:, 0], cand[:, 1])
    cand = cand[~inside]
    sdf = shapely.distance(boundary, shapely.points(cand[:, 0], cand[:, 1]))

    all_pos = np.vstack([surf_pts, cand])
    all_sdf = np.concatenate([np.zeros(n_surf), sdf])
    all_normals = np.vstack([surf_norm, np.zeros((len(cand), 2))])
    is_surface = np.concatenate([np.ones(n_surf, dtype=bool), np.zeros(len(cand), dtype=bool)])

    n_total = len(all_pos)
    inlet_velocity = np.tile([inlet_vx, inlet_vy], (n_total, 1))
    x = np.concatenate([all_pos, inlet_velocity, all_sdf[:, None], all_normals], axis=1).astype(np.float32)

    return {
        "x": x,
        "position": all_pos.astype(np.float32),
        "is_surface": is_surface,
        "surface_normals": all_normals[is_surface],
        "inlet_speed": float(inlet_speed),
        "aoa_rad": float(aoa_rad),
        "reynolds": float(reynolds),
        "aoa_deg": float(aoa_deg),
    }


def integrate_forces(surface_pos, surface_normal_vecs, pred_surface, offset_pred, eps, inlet_speed, aoa_rad, T=AIR_T):
    """Simplified panel-method force integration on the predicted surface field.

    surface_pos: (M, 2) node positions around the loop (closed, first==last not required here)
    surface_normal_vecs: (M, 2) INWARD unit normals at each node (into the solid,
        matching the AirfRANS training convention -- see surface_normals())
    pred_surface: (M, 4) predicted [vx, vy, p, nut] at the surface nodes
    offset_pred: (M, 4) predicted fields at surface_pos - eps*normal (just off the wall,
        i.e. offset outward into the fluid since the normal points inward)
    eps: the small normal offset distance used for the wall-shear finite difference
    """
    RHO = air_density(T)
    NU = air_kinematic_viscosity(T)

    M = surface_pos.shape[0]
    seg_len = np.linalg.norm(np.roll(surface_pos, -1, axis=0) - surface_pos, axis=1)
    panel_len = 0.5 * (seg_len + np.roll(seg_len, 1))

    pressure = pred_surface[:, 2]
    # normal is inward, so -p*n_outward = -p*(-n_inward) = +p*n_inward
    force_p_per_node = pressure[:, None] * surface_normal_vecs

    tangent = np.stack([-surface_normal_vecs[:, 1], surface_normal_vecs[:, 0]], axis=-1)
    du_dn = (offset_pred[:, :2] - pred_surface[:, :2]) / eps
    du_t_dn = np.sum(du_dn * tangent, axis=1)
    wss_per_node = NU * du_t_dn[:, None] * tangent

    Fp = np.sum(force_p_per_node * panel_len[:, None], axis=0) * RHO
    Fv = np.sum(wss_per_node * panel_len[:, None], axis=0) * RHO
    F = Fp + Fv

    basis = np.array([[np.cos(aoa_rad), np.sin(aoa_rad)], [-np.sin(aoa_rad), np.cos(aoa_rad)]])
    Fd, Fl = basis @ F
    q = 0.5 * RHO * inlet_speed**2
    return Fd / q, Fl / q


def check_envelope(reynolds, aoa_deg):
    warnings = []
    if not (ENVELOPE_RE[0] <= reynolds <= ENVELOPE_RE[1]):
        warnings.append(f"Reynolds {reynolds:.2e} is outside the training envelope {ENVELOPE_RE}")
    if not (ENVELOPE_AOA[0] <= aoa_deg <= ENVELOPE_AOA[1]):
        warnings.append(f"AoA {aoa_deg} deg is outside the training envelope {ENVELOPE_AOA}")
    return warnings
