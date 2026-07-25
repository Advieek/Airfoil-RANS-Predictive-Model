"""End-to-end prediction pipeline test: NACA code -> point cloud -> model
inference -> Cl/Cd. Adapted from scripts/step8_gate7_headless_test.py (the
original manual verification for this path) into an automated regression test.

Needs a real checkpoint, so it's skipped if the tracked checkpoints aren't
present (e.g. a shallow checkout without git-lfs/the checkpoints/ dir).
"""
import pathlib

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from src.app_core import load_geometry, predict_and_integrate  # noqa: E402
from src.evaluate import load_model_from_checkpoint  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CHECKPOINT = REPO_ROOT / "checkpoints" / "mlp_scarce_best.pt"
FULLRES_CHECKPOINT = REPO_ROOT / "checkpoints" / "mlp_full_fullres_v4_best.pt"

pytestmark = pytest.mark.skipif(not CHECKPOINT.exists(), reason="checkpoints/mlp_scarce_best.pt not present")


@pytest.fixture(scope="module")
def loaded_model():
    return load_model_from_checkpoint(str(CHECKPOINT), device="cpu")


@pytest.fixture(scope="module")
def fullres_model():
    return load_model_from_checkpoint(str(FULLRES_CHECKPOINT), device="cpu")


@pytest.mark.parametrize(
    "naca_code,aoa,reynolds",
    [
        ("0012", 0.0, 3_000_000),
        ("2412", 5.0, 4_000_000),
        ("4415", 8.0, 5_000_000),
    ],
)
def test_predict_and_integrate_runs_end_to_end(loaded_model, naca_code, aoa, reynolds):
    model, stats, _ckpt = loaded_model
    geo = load_geometry(naca_code, n_volume=3000)
    cloud, pred, cd, cl = predict_and_integrate(geo, reynolds, aoa, model, stats)

    assert pred.shape[1] == 4
    assert cloud["position"].shape[0] == pred.shape[0]
    assert np.isfinite(cd) and np.isfinite(cl)


@pytest.mark.skipif(not FULLRES_CHECKPOINT.exists(), reason="checkpoints/mlp_full_fullres_v4_best.pt not present")
def test_symmetric_airfoil_at_zero_aoa_gives_near_zero_lift(fullres_model):
    """NACA0012 (symmetric) at 0 deg AoA should predict Cl close to zero -- a
    physics sanity check independent of the benchmark's numeric error bars.
    Uses the full-resolution MLP (the site's actual default, see README's
    "Which model should I use?"), not mlp_scarce -- the scarce checkpoint is
    known to be considerably less accurate (0.830 vs. 0.276 Cl rel. error)
    and does not reliably pass this check."""
    model, stats, _ckpt = fullres_model
    geo = load_geometry("0012", n_volume=4000)
    _cloud, _pred, _cd, cl = predict_and_integrate(geo, reynolds=4_000_000, aoa_deg=0.0, model=model, stats=stats)
    assert abs(cl) < 0.1
