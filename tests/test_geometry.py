"""Unit tests for src/geometry.py: NACA generation, resampling, the inward-normal
convention, point-cloud construction, and the panel-method force integration.

No AirfRANS dataset or trained checkpoint needed -- everything here is pure
geometry/math on synthetic input, so it runs in CI with just the pinned deps.
"""
import numpy as np
import pytest

from src.geometry import (
    air_density,
    air_kinematic_viscosity,
    apply_inflow,
    check_envelope,
    generate_geometry_cloud,
    integrate_forces,
    naca_airfoil,
    resample_close,
    surface_normals,
)


def _circle(n=200, radius=0.5, cx=0.5):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + radius * np.cos(theta), radius * np.sin(theta)], axis=-1)


def test_naca_airfoil_returns_closed_ish_loop():
    coords = naca_airfoil("0012", nb_samples=100)
    assert coords.shape[1] == 2
    assert coords.shape[0] > 10
    # chord roughly spans [0, 1] before normalization
    assert coords[:, 0].max() - coords[:, 0].min() == pytest.approx(1.0, abs=0.05)


def test_naca_airfoil_rejects_bad_code_length():
    with pytest.raises(ValueError):
        naca_airfoil("12", nb_samples=50)


def test_resample_close_chord_normalizes_to_unit_x_range():
    raw = naca_airfoil("2412", nb_samples=150)
    out, chord = resample_close(raw, n_points=64)
    assert out.shape == (65, 2)  # n_points + closing point
    np.testing.assert_allclose(out[0], out[-1])
    # spline resampling doesn't necessarily land a sample exactly at the
    # chord extremes, so allow a small margin rather than requiring exact 0/1
    assert out[:, 0].min() == pytest.approx(0.0, abs=0.01)
    assert out[:, 0].max() == pytest.approx(1.0, abs=0.01)
    assert chord > 0


def test_surface_normals_point_inward():
    """The topmost point of a circle has an outward-pointing normal of (0, 1);
    this project's convention is inward, so it must come back as ~(0, -1) --
    the exact check that caught the original sign-convention bug (see
    src/geometry.py's surface_normals docstring)."""
    loop = _circle(n=400)
    loop = np.vstack([loop, loop[0]])
    normals = surface_normals(loop)
    top_idx = np.argmax(loop[:-1, 1])
    top_normal = normals[top_idx]
    np.testing.assert_allclose(top_normal, [0.0, -1.0], atol=0.05)


def test_surface_normals_are_unit_length():
    loop = _circle(n=100)
    loop = np.vstack([loop, loop[0]])
    normals = surface_normals(loop)
    lengths = np.linalg.norm(normals, axis=1)
    np.testing.assert_allclose(lengths, 1.0, atol=1e-6)


def test_generate_geometry_cloud_shapes_and_no_interior_volume_points():
    surface, _ = resample_close(naca_airfoil("0012", nb_samples=150), n_points=100)
    geo = generate_geometry_cloud(surface, n_volume=2000, seed=0)
    n_surf = surface.shape[0] - 1

    assert geo["is_surface"].sum() == n_surf
    assert geo["position"].shape[0] == geo["sdf"].shape[0] == geo["is_surface"].shape[0]
    # surface points have sdf == 0 by construction
    np.testing.assert_allclose(geo["sdf"][geo["is_surface"]], 0.0)
    # volume points must be strictly outside the airfoil (sdf > 0)
    assert (geo["sdf"][~geo["is_surface"]] > 0).all()


def test_apply_inflow_matches_reynolds_scaling():
    surface, _ = resample_close(naca_airfoil("0012", nb_samples=100), n_points=50)
    geo = generate_geometry_cloud(surface, n_volume=500, seed=0)
    cloud = apply_inflow(geo, reynolds=4e6, aoa_deg=0.0)

    nu = air_kinematic_viscosity()
    expected_speed = 4e6 * nu
    assert cloud["inlet_speed"] == pytest.approx(expected_speed)
    # AoA = 0 -> inflow purely in +x
    inlet_vx, inlet_vy = cloud["x"][0, 2], cloud["x"][0, 3]
    assert inlet_vx == pytest.approx(expected_speed, rel=1e-5)
    assert inlet_vy == pytest.approx(0.0, abs=1e-6)


def test_integrate_forces_symmetric_zero_pressure_gives_zero_force():
    """A symmetric point ring with uniform pressure and no shear should
    integrate to ~zero net force -- a sanity check on the panel-method math
    independent of any trained model."""
    n = 64
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    surface_pos = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    # inward normals for a circle centered at origin
    normals = -surface_pos.copy()
    pred_surface = np.zeros((n, 4), dtype=np.float32)  # zero velocity/pressure/nut
    offset_pred = np.zeros((n, 4), dtype=np.float32)  # no gradient -> no shear

    cd, cl = integrate_forces(
        surface_pos, normals, pred_surface, offset_pred, eps=0.01, inlet_speed=50.0, aoa_rad=0.0
    )
    assert cd == pytest.approx(0.0, abs=1e-8)
    assert cl == pytest.approx(0.0, abs=1e-8)


def test_check_envelope_flags_out_of_range_inputs():
    assert check_envelope(4e6, 5.0) == []
    warnings = check_envelope(1e6, 20.0)
    assert len(warnings) == 2


def test_air_density_and_viscosity_are_physically_reasonable():
    # air at ~25C: density ~1.18 kg/m^3, kinematic viscosity ~1.5e-5 m^2/s
    assert air_density() == pytest.approx(1.18, rel=0.05)
    assert air_kinematic_viscosity() == pytest.approx(1.5e-5, rel=0.15)
