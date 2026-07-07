# PROGRESS

## 2026-07-07

- [2026-07-07 initial] Repo initialized. git user set locally (adv / advieek@gmail.com).
- [2026-07-07] Step 0 bootstrap check: xcode CLT present (`/Library/Developer/CommandLineTools`), uv 0.11.27 present, git 2.50.1 present. Disk: 837Gi free. Dataset already unzipped at `data/Dataset` (1000 sim dirs + extra file, matches expected 1000 sims). No download needed.
- [2026-07-07] Step 1 done: `uv venv --python 3.11`, installed torch 2.12.1, torch_geometric 2.8.0, airfrans, pyvista, shapely, numpy, scipy, matplotlib, tensorboard, streamlit, netron, onnx, pillow. Lockfile: `requirements.txt` (uv pip freeze).
- **GATE 1 PASSED**: `torch.backends.mps.is_available() == True`; 4x4 matmul on `device='mps'` succeeded.
- Remember: export `PYTORCH_ENABLE_MPS_FALLBACK=1` for all training/eval runs.
- Next: Step 2/3 scaffold (models.py, dataset.py, train.py) against synthetic data, Gate 2.
