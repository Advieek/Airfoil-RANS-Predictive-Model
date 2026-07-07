"""Gate 7 headless test: exercise app.py's predict path (src.app_core) for 3
(airfoil, AoA, Re) combos with no Streamlit server running."""
import sys
import time

sys.path.insert(0, ".")
from src.app_core import load_geometry, predict_and_integrate
from src.evaluate import load_model_from_checkpoint

model, stats, ckpt = load_model_from_checkpoint("checkpoints/mlp_scarce_best.pt", device="cpu")

combos = [
    ("2412", 5.0, 4_000_000),
    ("0012", 0.0, 3_000_000),
    ("4415", 8.0, 5_000_000),
]

for naca_code, aoa, reynolds in combos:
    t0 = time.time()
    geo = load_geometry(naca_code, n_volume=6000)
    geo_time = time.time() - t0
    t1 = time.time()
    cloud, pred, cd, cl = predict_and_integrate(geo, reynolds, aoa, model, stats)
    predict_time = time.time() - t1
    print(
        f"NACA{naca_code} Re={reynolds:.1e} AoA={aoa}: Cl={cl:.4f} Cd={cd:.4f} "
        f"(geometry {geo_time:.3f}s, predict {predict_time:.3f}s)"
    )
    assert pred.shape[1] == 4
    assert cloud["position"].shape[0] == pred.shape[0]

print("GATE 7 HEADLESS TEST PASSED")
