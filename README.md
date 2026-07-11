# Airfoil RANS Surrogate

A neural network that takes a 2D airfoil geometry + Reynolds number + angle of
attack and predicts the RANS flow field (velocity, pressure, turbulent
viscosity) everywhere around the airfoil, then integrates the predicted
surface fields into lift/drag coefficients (Cl/Cd) — in milliseconds instead
of the minutes-to-hours a real OpenFOAM solve takes. Trained on
[AirfRANS](https://arxiv.org/abs/2212.07564) (1,000 real OpenFOAM k-ω SST
simulations, NeurIPS 2022 benchmark).

Built per `BUILD_SPEC_for_claude_code_v2.md` on a 16GB Mac mini, then scaled
up (full dataset, full mesh resolution, bigger models) on an M4 Pro / 64GB.
`PROGRESS.md` is the full timestamped build log — every step, every bug found
and how it was fixed, kept as-written rather than cleaned up in hindsight.

## Which model should I use?

Four checkpoints were trained (MLP and GraphSAGE, each on the `scarce` 200-sim
task and the `full` 800-sim task). **GraphSAGE (full task, 64k points/sim) is
the best overall choice for real predictions on real airfoil meshes.** It
wins on drag by a wide margin (8.40 vs. the MLP's 15.91 relative error, and
the only model besides scarce-GraphSAGE with a *positive* drag rank
correlation) while giving up almost nothing on lift (0.976 vs. 0.981 Spearman
— a rounding-level difference). A simple win-count across the four metrics
looks like a tie between the two full-task models, but weighting by how large
each gap actually is (min-max normalized composite score across all four
metrics) makes the winner clear:

| model | composite score | Cl rel. err | Cl Spearman | Cd rel. err | Cd Spearman |
|---|---|---|---|---|---|
| **GraphSAGE — full task** | **0.896** | 0.415 | 0.976 | **8.40** | **0.105** |
| MLP — full task | 0.732 | **0.276** | **0.981** | 15.91 | -0.125 |
| MLP — scarce task | 0.587 | 0.830 | 0.950 | 17.62 | -0.186 |
| GraphSAGE — scarce task | 0.250 | 2.815 | 0.804 | 44.07 | 0.251 |

This is what `web/api.py` marks `recommended` and what the Results page
highlights as best.

**Exception**: the interactive Tool page (both `web/` and `docs/`) defaults
to the **MLP**, not GraphSAGE, despite the above. GraphSAGE's accuracy is
tightly coupled to matching the point density it trained on (64k points/sim,
sampled from a real mesh); the Tool's arbitrary-airfoil point clouds are
synthetic and don't match that density, which measurably hurts it there
specifically — verified directly: GraphSAGE predicts Cl≈2.66 for NACA0012 at
3° AoA, where thin-airfoil theory gives 2π·sin(3°)≈0.33 and the MLP predicts
0.338. The MLP has no such dependency (no graph, purely per-point), so it's
the more *robust* choice for arbitrary shapes even though it's not the best
choice on the official benchmark. Both are still selectable from the
dropdown in `web/`'s Tool page.

## Quick start

Two ways to run the site — same design, different tradeoffs:

**`web/` — FastAPI, full-featured, needs a local server.**

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
export PYTORCH_ENABLE_MPS_FALLBACK=1   # Apple Silicon only
uvicorn web.api:app --port 8000
```

Open `http://localhost:8000`. Both trained models are selectable, predictions
run server-side (real PyTorch inference, not the ONNX export), and the Files
page browses the actual live filesystem with a sandboxed API (path traversal
blocked, `.venv`/`.git`/`data/Dataset` excluded from listings).

**`docs/` — fully static, zero backend, GitHub Pages-ready.**

```bash
cd docs && python3 -m http.server 8080
```

Open `http://localhost:8080`. The Tool page runs the MLP entirely in-browser
via [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) — no
server, no network call after the page loads. The whole geometry pipeline
(NACA generation, point-cloud construction, panel-method force integration)
is reimplemented in JavaScript (`docs/assets/js/geometry.js`) and was
verified against the Python original before being trusted: physical
constants match bit-for-bit, the (non-obvious — see below) inward-normal sign
convention matches, the force-integration math matches bit-for-bit on
synthetic test data, and an end-to-end prediction matches the server-side
result to three decimal places. Only the MLP runs client-side (GraphSAGE
would need k-NN graph construction ported too — not done, disclosed on the
page rather than silently omitted).

**To deploy `docs/` on GitHub Pages**: push this repo to GitHub, then in
Settings → Pages set the source to the `docs/` folder on your default branch.
First update the placeholder `REPO_URL` in `docs/files.html` to point at the
real repo.

## Full setup from scratch

```bash
git clone <this-repo> && cd airfoil-surrogate
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
export PYTORCH_ENABLE_MPS_FALLBACK=1   # Apple Silicon only; harmless elsewhere
```

That's it for using the trained models — the four canonical checkpoints
(`checkpoints/*_best.pt`, `norm_stats.json`, `model.onnx`, `eval_results_*.json`,
~3.3MB total) are tracked in git, so a fresh clone has working models
immediately. Verified directly: cloned the repo into a scratch directory and
ran `predict.py` and the full `web/` API against nothing but the tracked
files — both worked identically to the development copy.

You do **not** need the AirfRANS dataset itself unless you want to retrain or
re-run evaluation. If you do: it's a ~10GB download, see
`BUILD_SPEC_for_claude_code_v2.md`'s pre-flight section for the exact command
(downloads to `data/`, gitignored).

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
  export_onnx.py                # re-export a checkpoint to ONNX (single-file, for docs/)
  step*_gate*.py                 # one-off verification scripts from the original build
predict.py      # CLI: predict Cl/Cd/fields for a NACA code or .dat file
app.py          # Streamlit demo (superseded by web/, kept for quick local checks)
web/            # FastAPI multi-page site: templates, static assets, API
docs/           # static mirror of web/ for GitHub Pages -- geometry pipeline
                # re-implemented in JS, model runs client-side via ONNX Runtime Web
plots/          # sanity checks, dashboards, evolution GIFs
checkpoints/    # 4 canonical trained checkpoints tracked in git (see above);
                # everything else here (backups, resume state, intermediate
                # training checkpoints) is gitignored
runs/           # TensorBoard logs (gitignored)
```

## Usage

```bash
# multi-page website, full-featured (recommended)
uvicorn web.api:app --port 8000

# multi-page website, fully static (no backend)
cd docs && python3 -m http.server 8080

# CLI prediction on a known NACA shape
python predict.py --naca 2412 --reynolds 4e6 --aoa 5.0

# or an arbitrary Selig .dat file
python predict.py --dat foil.dat --reynolds 4e6 --aoa 5.0
# pick a specific checkpoint (defaults to the full-resolution MLP)
python predict.py --naca 2412 --reynolds 4e6 --aoa 5.0 \
  --checkpoint checkpoints/graphsage_full_64k_best.pt

# re-run evaluation on the withheld AirfRANS test set (needs the dataset)
python -m src.evaluate --checkpoint checkpoints/mlp_full_fullres_v4_best.pt \
  --out checkpoints/eval_results_mlp_full_fullres.json

# Streamlit demo (older, superseded by the website, kept for quick checks)
streamlit run app.py

# watch training live (while src/train.py is running)
tensorboard --logdir runs --port 6006

# view the model architecture graph
netron checkpoints/model.onnx
```

### Reproducing training

Needs the full AirfRANS dataset (`data/Dataset/`, see Full setup above).
Training is long-running (hours) and was, in practice, interrupted by a real
MPS memory leak during development — so it's wrapped in a supervisor that
auto-resumes from the last checkpoint on any crash rather than losing
progress:

```bash
# full task, full mesh resolution, MLP -- ~11-20h depending on hardware
./scripts/train_supervised.sh mlp_full_fullres 400 -- \
  --mode real --model mlp --task full --full-resolution --sims-per-batch 4 \
  --hidden 256,256,256,256,256 --rss-limit-gb 40 --checkpoint-every 10

# full task, 64k points/sim, GraphSAGE -- ~40h+
./scripts/train_supervised.sh graphsage_full_64k 400 -- \
  --mode real --model graphsage --task full --n-points 64000 --sims-per-batch 1 \
  --gnn-hidden 128 --gnn-layers 4 --gnn-k 10 --gnn-knn-backend cpu_kdtree \
  --rss-limit-gb 40 --checkpoint-every 10
```

`--rss-limit-gb` and `--checkpoint-every` are what make this safe to leave
unattended: if memory climbs past the limit, `src/train.py` exits cleanly
with a checkpoint already saved (not silently OOM-killed), and the supervisor
script relaunches with `--resume-from` automatically. Worst case after a
crash is losing `--checkpoint-every` epochs, not the whole run. See
`PROGRESS.md`'s 2026-07-08/09 entries for the incident that motivated this.

## How it works

See the website's **How it works** page for the full writeup (point-cloud
representation, both architectures, why training was hard to scale up). The
short version: every mesh point is an independent example with 7 input
features (position, inlet velocity, distance-to-surface, surface normal —
inward-pointing, a real and non-obvious AirfRANS convention) and 4 targets
(velocity, pressure, turbulent viscosity). The MLP predicts each point
independently; GraphSAGE adds message-passing over a k-nearest-neighbor graph
so each point has some awareness of its neighbors, which is what makes it
better at drag (a near-wall-gradient-dependent quantity a context-free
per-point model structurally can't resolve).

## Results (full AirfRANS dataset, full mesh resolution, 200-sim held-out test set)

| model | Cl rel. error | Cl Spearman | Cd rel. error | Cd Spearman |
|---|---|---|---|---|
| GraphSAGE (64k pts/sim) | 0.415 | 0.976 | **8.40** | **0.105** |
| MLP (full resolution) | **0.276** | **0.981** | 15.91 | -0.125 |
| GraphSAGE (scarce, 16k pts/sim) | 2.815 | 0.804 | 44.07 | 0.251 |
| MLP (scarce) | 0.830 | 0.950 | 17.62 | -0.186 |
| AirfRANS paper (MLP/scarce) | 0.385 | 0.981 | 3.50 | -0.139 |

Cl Spearman = rank correlation between predicted and true lift coefficient
across the 200 test simulations (1.0 = perfect ranking) — the standard
headline metric for this benchmark. Lift is well-predicted across the board;
drag remains the hard target for every model here, including the paper's own
baseline (it depends on near-wall velocity gradients that are a tiny
fraction of total field variance, so models optimize past them without a
specific inductive bias for it). Full breakdown, including a real evaluation
bug that initially corrupted these numbers and how it was found and fixed,
in `PROGRESS.md` (2026-07-11 entries) and on the website's Results page.

## Security notes

Reviewed before treating this as publishable, not just assumed clean:

- **No secrets in the repo** — searched for API keys, tokens, passwords,
  private-key markers; none found.
- **`web/api.py`'s file browser is sandboxed**: requested paths are resolved
  against the repo root and rejected if they escape it (tested directly —
  `../../../etc/passwd` returns 403); `.venv/`, `.git/`, `data/Dataset/`, and
  `.claude/` are excluded from listings regardless of path.
- **No secrets/eval/exec/shell injection** in the request-handling code path
  — inputs (`naca_code`, `dat_text`, file paths) only ever reach numeric
  parsing or the sandboxed path resolver above.
- **No CORS misconfiguration** — no CORS middleware is installed, so the API
  defaults to same-origin only.
- **XSS hardening**: every place the frontend JS renders dynamic content
  (warning messages, error text, filesystem names in the file browser) was
  switched from `innerHTML` template interpolation to DOM-API construction
  (`createElement`/`textContent`), so dynamic strings are never parsed as
  markup. Practical risk was low in every case found (single-user
  client-side tool, or content that was actually a server-controlled
  constant already) — fixed as a matter of not leaving the habit in a repo
  meant to be published, not because an exploit was demonstrated.
- **`docs/` is pure static assets** (HTML/CSS/JS/ONNX model/JSON) — no
  server-side code at all once deployed, so there's no request-handling
  surface to exploit in the GitHub Pages version.
- **Not reviewed / worth knowing if you expose `web/` beyond localhost**: it
  has no authentication, no rate limiting, and the file-download endpoint has
  no size cap (fine for local single-user use; add these before putting it
  on a shared or public-facing server). This is called out again under Next
  steps.

## Known limitations (see PROGRESS.md for detail)

- Cd prediction is unreliable relative to Cl for every checkpoint here —
  trust Cl more than Cd, and see "Which model should I use?" above for the
  best available tradeoff.
- `predict.py` / the website's Tool page, on arbitrary airfoils, use a
  simplified panel-method force integration (no mesh available for the exact
  wall-shear jacobian that the benchmark evaluation uses on real AirfRANS
  test sims), so Cd sign/magnitude on new shapes is noisier still.
- The static `docs/` build only runs the MLP; GraphSAGE needs a k-NN graph
  construction step not yet ported to JavaScript.
- No hyperparameter tuning, single seed per model — intentionally out of
  scope (see `BUILD_SPEC`'s "Scope discipline").

## Next steps

- **Fix GraphSAGE's point-density sensitivity for arbitrary airfoils** — the
  actual root cause behind why the Tool page defaults to the MLP. Likely fix:
  subsample the Tool's synthetic point clouds to match GraphSAGE's training
  density (64k pts) before inference, or train with randomized point density
  so the model generalizes across densities instead of overfitting to one.
- **Port GraphSAGE's k-NN construction to JavaScript** so the fully-static
  `docs/` build can offer both models, not just the MLP.
- **PointNet / Graph U-Net**, seed ensembles for uncertainty, `reynolds`/`aoa`
  extrapolation tasks, validation against XFOIL on non-NACA shapes — all
  explicitly deferred in the original build spec, still open.
- **Automated tests.** Everything so far is ad-hoc verification scripts
  (`scripts/step*_gate*.py`) run manually; a real `pytest` suite covering the
  data pipeline, force integration, and API endpoints would catch regressions
  the way this project's several manual audits caught them by hand.
- **If `web/` ever needs to run beyond localhost**: add authentication,
  rate-limiting, and a size cap on the file-download endpoint; put it behind
  a reverse proxy with TLS; reconsider whether the file browser should be
  exposed at all versus a curated download list.
- **Understand the MPS memory leak's actual root cause** rather than the
  current mitigation (periodic `empty_cache()` + a self-protecting RSS limit
  + auto-restart). The workarounds make it harmless, but the underlying
  cause in PyTorch's MPS allocator was never isolated.

## Acknowledgments

Trained on [AirfRANS](https://airfrans.readthedocs.io)
([paper](https://arxiv.org/abs/2212.07564), Bonnet et al., NeurIPS 2022
Datasets and Benchmarks Track). Not affiliated with the AirfRANS authors;
comparisons against their published baseline numbers are for context, not
official reproduction.
