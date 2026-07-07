# PROGRESS

## 2026-07-07

- [2026-07-07 initial] Repo initialized. git user set locally (adv / advieek@gmail.com).
- [2026-07-07] Step 0 bootstrap check: xcode CLT present (`/Library/Developer/CommandLineTools`), uv 0.11.27 present, git 2.50.1 present. Disk: 837Gi free. Dataset already unzipped at `data/Dataset` (1000 sim dirs + extra file, matches expected 1000 sims). No download needed.
- [2026-07-07] Step 1 done: `uv venv --python 3.11`, installed torch 2.12.1, torch_geometric 2.8.0, airfrans, pyvista, shapely, numpy, scipy, matplotlib, tensorboard, streamlit, netron, onnx, pillow. Lockfile: `requirements.txt` (uv pip freeze).
- **GATE 1 PASSED**: `torch.backends.mps.is_available() == True`; 4x4 matmul on `device='mps'` succeeded.
- Remember: export `PYTORCH_ENABLE_MPS_FALLBACK=1` for all training/eval runs.
- [2026-07-07] Verified actual airfrans column layout by reading installed package source (site-packages/airfrans/dataset.py `load()`): matches spec exactly — cols [0:2] position, [2:4] inlet velocity, [4:5] sdf, [5:7] normals (inputs, 7 total); [7:9] velocity, [9:10] pressure, [10:11] nu_t (targets, 4 total); [11] surface bool. Encoded as INPUT_COLS/TARGET_COLS/SURFACE_COL in src/dataset.py.
- [2026-07-07] Found `Simulation.force_coefficient()` in airfrans — its wall-shear computation needs the VTK mesh (jacobian via `compute_derivative`), so plan for Step 6: monkey-patch predicted velocity/pressure/nu_t onto a real `Simulation` object (same node order) and call its existing integration. For Step 7 (arbitrary airfoils, no mesh) will do a simpler panel integration over the parametrized surface curve instead.
- [2026-07-07] Wrote src/dataset.py, src/models.py (PerPointMLP + GraphSAGENet), src/train.py (synthetic + real modes, TensorBoard, evolution snapshots, CSV log, best-val checkpointing).
- **GATE 2 PASSED**: `python -m src.train --mode synthetic --epochs 3` on MPS — losses [4.339, 4.155, 3.776] (decreasing); TensorBoard event file written to runs/gate2_synthetic/; 3 evolution PNGs written to plots/evolution/.
- [2026-07-07] Step 4: added `load_scarce_cached` to src/dataset.py (memoizes float32 arrays to data/cache/*.pt via torch.save; raw airfrans load takes ~24s for 200 sims, cached load near-instant). Loaded scarce train split: 200 sims, shape (181794, 12) for sim 0 (float32 after cast), matches expected ~180k pts/sim. Split 180 train / 20 val (seed=0). Norm stats computed from the 180 train sims only, saved to checkpoints/norm_stats.json.
- Note on Gate 3 methodology: per-sim inlet velocity (x cols 2,3) is constant *within* one simulation, so a single-sim batch trivially shows std=0 on those two columns — expected, not a bug. Verified normalization correctness in aggregate over 30 sims instead: x mean≈[-0.01,0,0.05,-0.02,0.02,0,0], std≈[1.08,1.02,0.96,1.04,1.07,1.01,1.0]; y mean≈[0.14,0,0.01,0.02], std≈[1.04,1.05,0.96,1.02]. All within tolerance of (0,1).
- **GATE 3 PASSED**: single-sim normalized batch (32000, 7)/(32000,4) loads in 0.002s (cached) — well under 1s; aggregate mean/std close to (0,1); `plots/sanity_pressure_field.png` shows a clearly recognizable flow-around-airfoil pressure pattern (stagnation point, low-pressure suction peak). Script: scripts/step4_gate3_check.py.
- Next: Step 5 — train MLP with TensorBoard + evolution snapshots live, Gate 4.
