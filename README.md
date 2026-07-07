# Airfoil RANS Surrogate

A neural network that takes a 2D airfoil geometry + Reynolds number + angle of
attack and predicts the RANS flow field (velocity, pressure, turbulent
viscosity) everywhere around the airfoil, then integrates the predicted
surface fields into lift/drag coefficients (Cl/Cd). Trained on
[AirfRANS](https://arxiv.org/abs/2212.07564) (1,000 real OpenFOAM k-ω SST
simulations, NeurIPS 2022 benchmark), on Apple Silicon via PyTorch MPS.

Built per `BUILD_SPEC_for_claude_code_v2.md`; see `PROGRESS.md` for the full
timestamped build log, including two bugs found and fixed along the way
(a Gate-3 normalization-check false alarm, and a real inward-vs-outward
surface-normal sign bug caught at Gate 6).

## Setup

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Dataset: AirfRANS pre-processed `Dataset.zip`, unzipped to `data/Dataset/`
(gitignored). See `BUILD_SPEC_for_claude_code_v2.md` pre-flight for the
download command.

## Repository layout

```
src/
  dataset.py    # AirfRANS loading, train-only normalization, caching, subsampling
  models.py     # PerPointMLP, GraphSAGENet
  train.py      # training loop, TensorBoard logging, evolution snapshots
  evaluate.py   # chunked full-res inference, Cl/Cd via airfrans Simulation reuse
  geometry.py   # arbitrary-airfoil point clouds + panel-method force integration
  app_core.py   # non-UI prediction logic shared by app.py and its headless test
predict.py      # CLI: predict Cl/Cd/fields for a NACA code or .dat file
app.py          # Streamlit live demo
scripts/        # one-off gate-verification and utility scripts
plots/          # sanity checks, dashboards, evolution GIF
checkpoints/    # trained weights + norm stats (gitignored)
runs/           # TensorBoard logs (gitignored)
```

## Usage

```bash
# CLI prediction on a known NACA shape
python predict.py --naca 2412 --reynolds 4e6 --aoa 5.0

# or an arbitrary Selig .dat file
python predict.py --dat foil.dat --reynolds 4e6 --aoa 5.0

# live demo
streamlit run app.py

# watch training live (while src/train.py is running)
tensorboard --logdir runs --port 6006

# view the model architecture
netron checkpoints/model.onnx
```

## Results (MLP, `scarce` task, 200 train sims, 400 epochs)

| metric | ours | AirfRANS paper (MLP/scarce) |
|---|---|---|
| Cl relative error | 0.83 | 0.385 ± 0.097 |
| Cl Spearman | 0.950 | 0.981 ± 0.006 |
| Cd relative error | 17.6 | 3.50 ± 0.998 |
| Cd Spearman | -0.186 | -0.139 ± 0.185 |

Lift is well-predicted; drag is not — this matches the paper's own baseline
(a per-point MLP has no way to enforce spatial smoothness, so the near-wall
velocity gradients drag depends on come out noisy). See `PROGRESS.md` Step 6
and Step 7 for the full breakdown, and Step 9 for the GraphSAGE follow-up
(message passing should help specifically because it fixes this).

## Known limitations (see PROGRESS.md for detail)

- Cd prediction is unreliable for the MLP baseline (see above) — trust Cl,
  not Cd, from this checkpoint.
- `predict.py`/`app.py` on arbitrary airfoils use a simplified panel-method
  force integration (no mesh available for the exact wall-shear jacobian
  that Step 6 uses on real AirfRANS test sims), so Cd sign/magnitude on new
  shapes is noisier still.
- Trained on `scarce` (200 sims) only, one seed, no hyperparameter tuning —
  intentionally out of scope for a one-day build (see `BUILD_SPEC`'s "Scope
  discipline").

## If a GraphSAGE run finished overnight

Run `./morning_after.sh` — it re-evaluates on the test set, regenerates the
evolution GIF, and switches `app.py`/`predict.py` to the new checkpoint.
