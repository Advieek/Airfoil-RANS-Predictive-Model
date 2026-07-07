# BUILD SPEC v2 — Airfoil RANS Surrogate Neural Network
### For autonomous execution by Claude Code (Opus 4.8) on a FRESH Mac

> **How to use this file (instructions for the human):**
> 1. Do the pre-flight below (10 min of your time; the dataset download runs unattended).
> 2. Put this file in `~/airfoil-surrogate`, open Terminal there, run `claude`.
> 3. Prompt: *"Read BUILD_SPEC_for_claude_code_v2.md and execute it end to end. Work autonomously, follow the verification gates, keep PROGRESS.md updated."*
> 4. Press Shift+Tab to enable auto-accept edits (fewer interruptions = fewer wasted tokens). Approve shell commands as they come.

---

## HUMAN PRE-FLIGHT (saves hours of agent time and a lot of tokens)

The Mac is assumed fresh: browser + Claude app + Claude Code only. Do these in order; 1 and 2 **require a human** (GUI dialog / new shell), 3 just saves time. **Do NOT install Homebrew, Python, or Xcode — none are needed.**

1. **Xcode Command Line Tools** (git + compilers): run `xcode-select --install` in Terminal and click "Install" in the dialog (~5 min).
2. **uv** (Python manager, no admin password): `curl -LsSf https://astral.sh/uv/install.sh | sh` — then open a NEW Terminal window.
3. **Start the dataset download now** (~10 GB; runs while you do everything else; needs ≥ 40 GB free disk):
   ```bash
   mkdir -p ~/airfoil-surrogate/data && cd ~/airfoil-surrogate/data
   curl -L -C - -O https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip
   unzip -q Dataset.zip
   ```
   (`-C -` makes it resumable if interrupted — just rerun the same command.)
4. Put this spec file in `~/airfoil-surrogate/`, then launch Claude Code from that folder.

During the run, two things happen in YOUR browser, not the agent's: watching live training at http://localhost:6006 (TensorBoard) and the demo app at http://localhost:8501 (Streamlit). The agent starts the servers; you look at them.

---

## Instructions for the agent (Claude Code / Opus 4.8)

You are building a complete, working ML pipeline in ONE working day on a fresh Mac. Work autonomously: write code, run it, debug failures yourself, commit to git after each passing verification gate. Maintain `PROGRESS.md` (terse, timestamped: done / running / measured results). Only stop to ask the user before launching jobs expected to exceed ~2 hours.

### Token-efficiency rules (follow strictly)

- Run anything long via `nohup ... > logfile 2>&1 &` and poll with `tail -n 20`; never stream training output into context.
- Never `cat` data files or print arrays beyond `.shape`, dtype, and a 5-row head.
- Keep PROGRESS.md and commit messages terse. No prose recaps of code you just wrote.
- Check whether pre-flight artifacts already exist before creating them (CLT, uv, dataset).

### Mission

Input = 2D airfoil geometry + Reynolds number + angle of attack; output = predicted RANS flow fields (velocity, pressure, turbulent viscosity), integrated into lift/drag coefficients. Train on the public AirfRANS dataset. Deliverables: a CLI `predict.py`, a live Streamlit demo, TensorBoard-instrumented training, a training-evolution animation, and an ONNX export viewable in Netron.

### Hard constraints

- Apple Silicon Mac mini, 16 GB unified memory, PyTorch **MPS**. Export `PYTORCH_ENABLE_MPS_FALLBACK=1` for every run.
- **float32 everywhere** (airfrans data loads float64 — cast at load; never float64 on MPS).
- Batch = 1 simulation; subsample ≤ 32,000 points/sim/epoch in training; full resolution only at evaluation, chunked.
- Download ONLY the pre-processed dataset — never the raw OpenFOAM version (`OpenFOAM=False`).
- Fix seeds; save normalization stats alongside every checkpoint; gitignore `data/`, `checkpoints/`, `runs/`.
- Normalization statistics from the training split ONLY.

### Dataset knowledge (verified facts — rely on these)

- `pip install airfrans`; docs https://airfrans.readthedocs.io
- Download (if pre-flight didn't): `airfrans.dataset.download(root="data/", file_name="Dataset", unzip=True, OpenFOAM=False)` — equivalent to the curl in pre-flight.
- Load: `airfrans.dataset.load(root="data/Dataset", task="scarce", train=True)` → (list of per-sim float64 arrays, list of names). **Inspect the actual column layout at runtime and record it in PROGRESS.md before building the pipeline.** Expected per point: position (2), inlet velocity (2), distance function (1), normals (2) as inputs; velocity (2), pressure (1), turbulent viscosity (1) as targets; plus surface boolean.
- 1,000 RANS sims (OpenFOAM k-ω SST), NACA 4/5-digit, Re 2–6M, AoA −5°…15°, ~180k nodes each, domain x ∈ (−2, 4), y ∈ (−1.5, 1.5).
- Tasks: `full` (800/200), `scarce` (200/200 — **use this today**), `reynolds`, `aoa`.
- `airfrans.Simulation(root, name)` has force-coefficient computation — read its source in site-packages and reuse the integration on predicted fields.
- `airfrans.naca_generator` generates NACA coordinates (used in Steps 8–9).
- Reference baselines + hyperparameters: https://github.com/Extrality/AirfRANS (CUDA-targeted; adapt device handling to MPS). Alternative loader: `torch_geometric.datasets.AirfRANS`.

### Repository layout to produce

```
airfoil-surrogate/
├── BUILD_SPEC_for_claude_code_v2.md
├── PROGRESS.md
├── requirements lockfile (uv)
├── data/            (gitignored)   ├── checkpoints/ (gitignored)
├── runs/            (gitignored)   ├── plots/  └── plots/evolution/
├── src/
│   ├── dataset.py   # load, float32, normalize, subsample
│   ├── models.py    # MLP + GraphSAGE
│   ├── train.py     # loop + TensorBoard + evolution snapshots
│   ├── evaluate.py  # chunked full-res inference, forces, metrics
│   └── geometry.py  # arbitrary-airfoil point clouds (Step 7)
├── app.py           # Streamlit live demo
└── predict.py       # CLI
```

---

## Execution plan (one day, in order)

### Step 0 — Bootstrap check (≤ 10 min)

Verify: `xcode-select -p` succeeds (else run `xcode-select --install`, tell the user to click the dialog, and wait); `uv --version` works (else install via the curl above); dataset presence at `data/Dataset` (else start the download in the background NOW and reorder work around it); free disk ≥ 40 GB. Set git user config if unset.

### Step 1 — Environment (≤ 30 min)

`uv venv --python 3.11` (uv fetches Python itself); install: `torch torch_geometric airfrans pyvista shapely numpy scipy matplotlib tensorboard streamlit netron onnx pillow`.

**GATE 1:** script prints `torch.backends.mps.is_available() == True` and multiplies two matrices on `device="mps"`. Commit.

### Step 2 — Dataset ready?

If pre-flight download finished: verify the unzipped layout (list top-level of `data/Dataset`, count sims ≈ 1000). If still running: continue to Step 3; poll.

### Step 3 — Scaffold while anything downloads (~1 h)

Write `models.py`, `dataset.py`, `train.py` against the documented API. Include from the start: TensorBoard `SummaryWriter` logging and an `--evolution-sim` option that snapshots one fixed validation airfoil's predicted pressure field every 10 epochs to `plots/evolution/epoch_XXXX.png` — **fixed colormap and fixed color limits across all frames** (compute limits from ground truth once). Prove the loop on synthetic data.

**GATE 2:** 3 synthetic epochs run on MPS, loss decreases, a TensorBoard event file appears, one snapshot PNG renders. Commit.

### Step 4 — Data pipeline on real data (~1 h)

Load `scarce` train split; record actual shapes/columns in PROGRESS.md. Hold out 20/200 sims for validation; test set untouched until Step 6. Compute train-only mean/std → `checkpoints/norm_stats.json`. Cache float32 tensors. Save one sanity scatter (pressure-colored) to `plots/`.

**GATE 3:** DataLoader yields normalized batch (mean≈0, std≈1) in < 1 s; sanity plot looks like flow around an airfoil. Commit.

### Step 5 — Train MLP, watched live (~2 h wall clock)

- Per-point MLP: 7 → [128,128,128,128] ReLU → 4 (~50k params). Adam 1e-3, cosine decay.
- 10-epoch smoke run first. Then the real run (~400 epochs, `scarce`) in the background with `caffeinate`, logging every epoch to TensorBoard + CSV, evolution snapshot every 10 epochs.
- **Launch `tensorboard --logdir runs --port 6006` in the background and tell the user:** "Training is live — open http://localhost:6006 to watch." 
- Save best-val checkpoint + norm stats.

**GATE 4:** val-loss curve in TensorBoard clearly decreasing; `plots/evolution/` frames get visibly sharper over epochs; side-by-side predicted-vs-true pressure plot for one val sim is a recognizable flow field. Commit; record final losses.

### Step 6 — Force-based evaluation (~1–2 h)

Chunked full-resolution inference on all test sims → un-normalize → Cl/Cd via the airfrans-derived integration. Report in PROGRESS.md: per-field MSE (volume + surface), mean relative error on Cl and Cd, **Spearman rank correlation** for Cl and Cd. Compare magnitudes against the AirfRANS paper's MLP/scarce row. Plots: predicted-vs-true scatter (Cl, Cd), error contour of worst test airfoil.

**GATE 5:** metrics table exists; Cl Spearman is high (near-zero means a bug — check normalization or integration first). Commit.

### Step 7 — Arbitrary-airfoil inference (~2 h)

`geometry.py` + `predict.py`:
1. Parse Selig `.dat` files and raw (x,y) arrays; spline-resample, chord-normalize, close trailing edge.
2. Generate point cloud matching the training distribution: dense near surface within x ∈ (−2, 4), y ∈ (−1.5, 1.5) (sample the density profile from a real training sim); drop interior points (shapely); compute SDF + normals; inflow features from Re/AoA.
3. Predict → un-normalize → integrate → print Cl/Cd; save Cp + field plots. WARN when inputs leave the training envelope.
4. CLI: `python predict.py --dat foil.dat --reynolds 4e6 --aoa 5.0 [--checkpoint ...]`

**GATE 6:** generate NACA 2412 via `airfrans.naca_generator`, run at Re 4e6 / AoA 5°: Cl plausibly ~0.5–1.0, Cd order 0.01, field plots sensible. Commit.

### Step 8 — Visualization suite (~1.5 h)  ← v2 addition

1. **Training-evolution GIF:** stitch `plots/evolution/*.png` with pillow, ~15 fps, epoch number annotated per frame → `plots/training_evolution.gif`. This is the "network learning physics from noise" movie.
2. **ONNX + Netron:** export the trained model on **CPU** (`model.to("cpu")` before `torch.onnx.export`) → `checkpoints/model.onnx`. Verify `netron checkpoints/model.onnx` serves without error, then stop it and record the command in PROGRESS.md for the user.
3. **Streamlit live demo (`app.py`):**
   - Sidebar: airfoil source (NACA 4-digit code via `naca_generator`, or `.dat` upload), AoA slider (−5…15°, step 0.5), Re slider (2e6…6e6).
   - Main: predicted pressure or velocity field (point scatter, fixed colorbar), big Cl/Cd readouts, envelope warning banner when applicable.
   - Performance: `@st.cache_resource` for model, `@st.cache_data` for the per-airfoil point cloud. Slider moves only change the two inflow-feature columns → one forward pass per update, sub-second. Recompute geometry only when the airfoil changes.
4. **Optional if time remains:** hidden-layer view — forward hook on a middle layer, color points by one activation channel, add as a toggle in the app.

**GATE 7:** GIF plays and shows sharpening; scripted headless test of `app.py`'s predict path passes for 3 (airfoil, AoA, Re) combos; `streamlit run app.py` starts cleanly — tell the user: "Demo is live at http://localhost:8501 — drag the AoA slider." Commit.

### Step 9 — GraphSAGE upgrade (launch today, finishes overnight)

- GraphSAGE in `models.py`: `knn_graph` (k≈8–16) per batch on subsampled points; hyperparameters adapted from the Extrality repo; same TensorBoard + evolution instrumentation (separate run name so curves overlay the MLP's).
- 10-epoch smoke run (drop to 10–16k points/sim if memory pressure is high). **Ask the user**, then launch the full run in background with `caffeinate`.
- Write `morning_after.sh`: re-runs Step 6 evaluation, regenerates the evolution GIF, and points `app.py` + `predict.py` at the new checkpoint.

**GATE 8:** smoke losses logged; long run launched; PROGRESS.md ends with exact morning-after instructions. Final commit + brief README.

---

## End-of-day deliverables checklist

- [ ] Git repo, ≥ 8 milestone commits, brief README
- [ ] Trained MLP checkpoint + norm stats
- [ ] PROGRESS.md metrics table (field MSE, Cl/Cd rel. error, Spearman)
- [ ] `predict.py` demonstrated on NACA 2412
- [ ] TensorBoard runs for all training; `plots/training_evolution.gif`
- [ ] `checkpoints/model.onnx` + Netron command recorded
- [ ] `app.py` live demo working with sub-second slider response
- [ ] GraphSAGE training running overnight + `morning_after.sh`

## Known failure modes (check FIRST when debugging)

1. float64 tensor reaches MPS → cast at load.
2. Test-set leakage into norm stats → recompute from train only.
3. Forgot to un-normalize before force integration → absurd Cl/Cd.
4. Evolution frames with per-frame color scaling → animation shows nothing; fix limits once.
5. PyG scatter ops CPU-fallback → slow but correct; note it, don't fight it today.
6. ONNX export on MPS device → move model to CPU first.
7. Streamlit recomputing geometry every slider tick → cache; only inflow columns change.
8. Assumed (not inspected) airfrans column order → garbage training; inspect in Step 4.
9. Mac sleeps mid-run → `caffeinate`.

## Scope discipline

NOT today: PointNet/Graph U-Net, `full`/`reynolds`/`aoa` tasks, hyperparameter sweeps, app styling/polish beyond function, uncertainty ensembles, own CFD runs, activation view if time is short. List them under "Next steps" in PROGRESS.md. A finished, instrumented, demo-able MLP pipeline beats an unfinished fancy one.
