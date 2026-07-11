# Airfoil RANS Surrogate

A neural network that takes a 2D airfoil geometry + Reynolds number + angle of
attack and predicts the RANS flow field (velocity, pressure, turbulent
viscosity) everywhere around the airfoil, then integrates the predicted
surface fields into lift/drag coefficients (Cl/Cd). Trained on
[AirfRANS](https://arxiv.org/abs/2212.07564) (1,000 real OpenFOAM k-ω SST
simulations, NeurIPS 2022 benchmark).

Built per `BUILD_SPEC_for_claude_code_v2.md` on a 16GB Mac mini, then scaled
up (full dataset, full mesh resolution, bigger models) on an M4 Pro / 64GB.
See `PROGRESS.md` for the full timestamped build log, including several real
bugs found and fixed along the way (a surface-normal sign error, an MPS
memory leak that cost a day of training before a resumable-checkpoint +
auto-restart system fixed it for good, and a chunked-evaluation bug that
silently corrupted GraphSAGE's benchmark numbers).

## Website

```bash
uv pip install -r requirements.txt
uvicorn web.api:app --port 8000
```

Open `http://localhost:8000` for the multi-page site: project overview,
methodology writeup, full results comparison, a live interactive prediction
tool, and a browser over the repo itself (code, checkpoints, plots — all
downloadable). This is the primary way to use the project now; see
`web/api.py` for the small FastAPI backend behind it.

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
  models.py     # PerPointMLP, GraphSAGENet (+ fast GPU/CPU k-NN backends)
  train.py      # training loop, TensorBoard logging, evolution snapshots,
                # resumable checkpoints, RSS-limit self-guard
  evaluate.py   # full-res inference, Cl/Cd via airfrans Simulation reuse
  geometry.py   # arbitrary-airfoil point clouds + panel-method force integration
  app_core.py   # non-UI prediction logic shared by web/api.py, app.py, predict.py
scripts/
  train_supervised.sh          # auto-resuming training supervisor
predict.py      # CLI: predict Cl/Cd/fields for a NACA code or .dat file
app.py          # Streamlit demo (superseded by web/, kept for quick local checks)
web/            # FastAPI multi-page site: templates, static assets, API
plots/          # sanity checks, dashboards, evolution GIFs
checkpoints/    # trained weights + norm stats + eval results (gitignored)
runs/           # TensorBoard logs (gitignored)
```

## Usage

```bash
# multi-page website (recommended)
uvicorn web.api:app --port 8000

# CLI prediction on a known NACA shape
python predict.py --naca 2412 --reynolds 4e6 --aoa 5.0

# or an arbitrary Selig .dat file
python predict.py --dat foil.dat --reynolds 4e6 --aoa 5.0

# Streamlit demo (older, superseded by the website)
streamlit run app.py

# watch training live (while src/train.py is running)
tensorboard --logdir runs --port 6006

# view the model architecture
netron checkpoints/model.onnx
```

## Results (full AirfRANS dataset, full mesh resolution, 200-sim held-out test set)

| model | Cl rel. error | Cl Spearman | Cd rel. error | Cd Spearman |
|---|---|---|---|---|
| GraphSAGE (64k pts/sim) | 0.415 | **0.976** | 8.40 | 0.105 |
| MLP (full resolution) | **0.276** | 0.981 | 15.91 | -0.125 |
| GraphSAGE (scarce, 16k pts/sim) | 2.815 | 0.804 | 44.07 | 0.251 |
| MLP (scarce) | 0.830 | 0.950 | 17.62 | -0.186 |
| AirfRANS paper (MLP/scarce) | 0.385 | 0.981 | 3.50 | -0.139 |

Lift is well-predicted across the board; drag remains the hard target (it
depends on near-wall velocity gradients that are a tiny fraction of total
field variance, so models optimize past them). GraphSAGE's message passing
helps close some of that gap over the plain MLP. Full breakdown, including
the bug that initially corrupted these numbers and how it was found and
fixed, in `PROGRESS.md` (2026-07-11 entries) and on the website's Results
page.

**Note**: the interactive tool defaults to the MLP checkpoint, not the
GraphSAGE model that wins above — GraphSAGE's accuracy is tightly coupled to
matching its training-time point density, and the tool's synthetic point
clouds for arbitrary airfoils don't match any AirfRANS mesh density. The MLP
has no such dependency and is far more robust there. See the Tool page for
detail, or `web/api.py`'s `MODELS` registry.

## Known limitations (see PROGRESS.md for detail)

- Cd prediction is unreliable relative to Cl for every checkpoint here —
  trust Cl more than Cd.
- `predict.py` / the website's Tool page, on arbitrary airfoils, use a
  simplified panel-method force integration (no mesh available for the exact
  wall-shear jacobian that the benchmark evaluation uses on real AirfRANS
  test sims), so Cd sign/magnitude on new shapes is noisier still.
- No hyperparameter tuning, single seed per model — intentionally out of
  scope (see `BUILD_SPEC`'s "Scope discipline").
