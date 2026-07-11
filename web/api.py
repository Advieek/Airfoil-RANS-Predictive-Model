"""FastAPI backend for the Airfoil RANS Surrogate website: serves the
multi-page static site (Jinja2 templates) and a small JSON API backing the
interactive Tool page and the repo file browser.

Run from the repo root: uvicorn web.api:app --port 8000 --reload
"""
import hashlib
import json
import pathlib
import sys

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
WEB_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.app_core import load_geometry, predict_and_integrate  # noqa: E402
from src.evaluate import load_model_from_checkpoint  # noqa: E402
from src.geometry import check_envelope  # noqa: E402

app = FastAPI(title="Airfoil RANS Surrogate")
app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")
app.mount("/plots", StaticFiles(directory=str(REPO_ROOT / "plots")), name="plots")
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = [
    {
        "id": "graphsage_full_64k",
        "label": "GraphSAGE — full task, 64k pts/sim",
        "recommended": True,  # best on the official AirfRANS test-set benchmark (Results page)
        "tool_default": False,
        "checkpoint": "checkpoints/graphsage_full_64k_best.pt",
        "eval": "checkpoints/eval_results_graphsage_full_64k.json",
    },
    {
        "id": "mlp_full_fullres_v4",
        "label": "MLP — full task, full resolution",
        "recommended": False,
        "tool_default": True,  # best for THIS tool's synthetic point clouds -- see note below
        "checkpoint": "checkpoints/mlp_full_fullres_v4_best.pt",
        "eval": "checkpoints/eval_results_mlp_full_fullres.json",
    },
    {
        "id": "graphsage_scarce",
        "label": "GraphSAGE — scarce task, 16k pts/sim",
        "recommended": False,
        "tool_default": False,
        "checkpoint": "checkpoints/graphsage_scarce_best.pt",
        "eval": "checkpoints/eval_results_graphsage.json",
    },
    {
        "id": "mlp_scarce",
        "label": "MLP — scarce task",
        "recommended": False,
        "tool_default": False,
        "checkpoint": "checkpoints/mlp_scarce_best.pt",
        "eval": "checkpoints/eval_results_mlp.json",
    },
]
# Why the Tool page's default differs from the Results page's "best" model:
# GraphSAGE's accuracy is tightly coupled to matching its training-time k-NN
# point density (this is literally what the 2026-07-11 evaluation-bug
# investigation found). The Tool page generates a synthetic point cloud for
# arbitrary airfoils -- a different density/structure than any AirfRANS mesh
# -- and GraphSAGE predictions degrade badly on it (verified: NACA0012 @
# Re=3e6/AoA=3 deg gives Cl=2.66 from GraphSAGE vs. Cl=0.338 from the MLP,
# where thin-airfoil theory gives 2*pi*sin(3deg)=0.33 -- the MLP is right and
# GraphSAGE is badly wrong here). The MLP has no such dependency (no graph,
# purely per-point), so it generalizes far more robustly to this pipeline's
# synthetic point clouds, even though GraphSAGE wins decisively on the real
# AirfRANS test-set benchmark shown on the Results page.
MODEL_BY_ID = {m["id"]: m for m in MODELS}

PAPER_BASELINE = {
    "label": "AirfRANS paper (MLP / scarce)",
    "cl_rel_err": 0.385,
    "cl_spearman": 0.981,
    "cd_rel_err": 3.50,
    "cd_spearman": -0.139,
}


def _eval_summary(model_info):
    eval_path = REPO_ROOT / model_info["eval"]
    if not eval_path.exists():
        return None
    with open(eval_path) as f:
        d = json.load(f)
    return {
        "n_test_sims": d.get("n_test_sims"),
        "cl_rel_err": d.get("cl_rel_err"),
        "cl_spearman": d.get("cl_spearman"),
        "cd_rel_err": d.get("cd_rel_err"),
        "cd_spearman": d.get("cd_spearman"),
    }


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

PAGES = [
    ("/", "index.html", "Home"),
    ("/how-it-works", "how-it-works.html", "How it works"),
    ("/results", "results.html", "Results"),
    ("/tool", "tool.html", "Tool"),
    ("/files", "files.html", "Files"),
]


def _nav(active):
    return [{"href": href, "label": label, "active": href == active} for href, _, label in PAGES]


for _route, _template, _label in PAGES:

    def _make_handler(route=_route, template=_template):
        def handler(request: Request):
            return templates.TemplateResponse(request, template, {"nav": _nav(route)})

        return handler

    app.get(_route)(_make_handler())


# ---------------------------------------------------------------------------
# /api/models, /api/results
# ---------------------------------------------------------------------------


@app.get("/api/models")
def api_models():
    out = []
    for m in MODELS:
        out.append(
            {
                "id": m["id"],
                "label": m["label"],
                "recommended": m["recommended"],
                "tool_default": m["tool_default"],
                "metrics": _eval_summary(m),
            }
        )
    return out


@app.get("/api/results")
def api_results():
    return {"models": api_models(), "paper_baseline": PAPER_BASELINE}


# ---------------------------------------------------------------------------
# /api/predict
# ---------------------------------------------------------------------------

_model_cache = {}
_geometry_cache = {}


def get_model(model_id):
    if model_id not in MODEL_BY_ID:
        raise HTTPException(400, f"unknown model_id '{model_id}'")
    if model_id not in _model_cache:
        info = MODEL_BY_ID[model_id]
        ckpt_path = REPO_ROOT / info["checkpoint"]
        if not ckpt_path.exists():
            raise HTTPException(404, f"checkpoint not found for model '{model_id}'")
        _model_cache[model_id] = load_model_from_checkpoint(str(ckpt_path), device="cpu")
    return _model_cache[model_id]


def get_geometry_cached(source, naca_code, dat_text):
    if source == "naca":
        code = (naca_code or "").strip()
        key = ("naca", code)
        dat_bytes, airfoil_key = None, code
    elif source == "dat":
        dat_bytes = (dat_text or "").encode("utf-8")
        key = ("dat", hashlib.sha256(dat_bytes).hexdigest())
        airfoil_key = "uploaded"
    else:
        raise HTTPException(400, "source must be 'naca' or 'dat'")
    if key not in _geometry_cache:
        _geometry_cache[key] = load_geometry(airfoil_key, dat_bytes=dat_bytes)
    return _geometry_cache[key]


class PredictRequest(BaseModel):
    model_id: str
    source: str
    naca_code: str | None = None
    dat_text: str | None = None
    reynolds: float
    aoa: float


@app.post("/api/predict")
def api_predict(req: PredictRequest):
    warnings = list(check_envelope(req.reynolds, req.aoa))
    if req.source == "dat":
        warnings.append(
            "Arbitrary .dat geometry -- accuracy degrades outside the NACA 4/5-digit family the model trained on."
        )

    try:
        geo = get_geometry_cached(req.source, req.naca_code, req.dat_text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"failed to parse airfoil geometry: {e}")

    model, stats, ckpt = get_model(req.model_id)
    cloud, pred, cd, cl = predict_and_integrate(geo, req.reynolds, req.aoa, model, stats)

    def r(arr, nd=4):
        return np.round(np.asarray(arr), nd).tolist()

    return {
        "cl": float(cl),
        "cd": float(cd),
        "warnings": warnings,
        "position": r(cloud["position"]),
        "pressure": r(pred[:, 2]),
        "velocity": r(pred[:, :2]),
    }


# ---------------------------------------------------------------------------
# /api/files -- sandboxed repo browser
# ---------------------------------------------------------------------------

EXCLUDED_PREFIXES = [".venv", ".git", "__pycache__", "data/Dataset", ".claude", "node_modules"]
MAX_PREVIEW_BYTES = 512_000


def is_excluded(rel_path: str) -> bool:
    norm = rel_path.strip("/")
    for prefix in EXCLUDED_PREFIXES:
        if norm == prefix or norm.startswith(prefix + "/"):
            return True
    return False


def resolve_safe_path(rel_path: str) -> pathlib.Path:
    rel_path = (rel_path or "").lstrip("/")
    candidate = (REPO_ROOT / rel_path).resolve()
    root_str = str(REPO_ROOT)
    if candidate != REPO_ROOT and not str(candidate).startswith(root_str + "/"):
        raise HTTPException(403, "path escapes repository root")
    return candidate


@app.get("/api/files")
def api_files(path: str = ""):
    if is_excluded(path):
        raise HTTPException(404, "not found")
    target = resolve_safe_path(path)
    if not target.exists():
        raise HTTPException(404, "not found")

    if target.is_dir():
        entries = []
        for entry in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            rel = str(entry.relative_to(REPO_ROOT))
            if is_excluded(rel):
                continue
            entries.append(
                {
                    "name": entry.name,
                    "path": rel,
                    "is_dir": entry.is_dir(),
                    "size": entry.stat().st_size if entry.is_file() else None,
                }
            )
        return {"type": "dir", "path": path, "entries": entries}

    size = target.stat().st_size
    if size > MAX_PREVIEW_BYTES:
        return {"type": "file", "path": path, "size": size, "too_large": True}
    try:
        content = target.read_text(encoding="utf-8")
        return {"type": "file", "path": path, "size": size, "too_large": False, "binary": False, "content": content}
    except UnicodeDecodeError:
        return {"type": "file", "path": path, "size": size, "too_large": False, "binary": True}


@app.get("/api/files/download")
def api_files_download(path: str):
    if is_excluded(path):
        raise HTTPException(404, "not found")
    target = resolve_safe_path(path)
    if not target.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(str(target), filename=target.name)
