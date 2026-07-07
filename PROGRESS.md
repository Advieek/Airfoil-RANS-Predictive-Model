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
- Next: Step 4 — real data pipeline (load scarce split, train/val split, norm stats, sanity plot), Gate 3.
