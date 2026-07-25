"""Smoke tests for web/api.py: page routes, the /api/models and /api/predict
JSON endpoints, and the sandboxed file browser's path-traversal guard (the
thing the README's Security notes section claims is tested)."""
import pathlib

import pytest

pytest.importorskip("torch")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from web.api import app  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
HAS_CHECKPOINTS = (REPO_ROOT / "checkpoints" / "mlp_scarce_best.pt").exists()

client = TestClient(app)


@pytest.mark.parametrize("path", ["/", "/how-it-works", "/results", "/tool", "/files"])
def test_pages_return_200(path):
    resp = client.get(path)
    assert resp.status_code == 200


def test_api_models_lists_all_four_checkpoints():
    resp = client.get("/api/models")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 4
    ids = {m["id"] for m in body}
    assert ids == {"graphsage_full_64k", "mlp_full_fullres_v4", "graphsage_scarce", "mlp_scarce"}


def test_api_results_includes_paper_baseline():
    resp = client.get("/api/results")
    assert resp.status_code == 200
    assert "paper_baseline" in resp.json()


@pytest.mark.skipif(not HAS_CHECKPOINTS, reason="checkpoints not present")
def test_api_predict_naca_smoke():
    resp = client.post(
        "/api/predict",
        json={
            "model_id": "mlp_scarce",
            "source": "naca",
            "naca_code": "0012",
            "reynolds": 3_000_000,
            "aoa": 2.0,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "cl" in body and "cd" in body
    assert len(body["position"]) == len(body["pressure"])


def test_api_predict_rejects_unknown_model_id():
    resp = client.post(
        "/api/predict",
        json={"model_id": "nonexistent", "source": "naca", "naca_code": "0012", "reynolds": 3e6, "aoa": 0.0},
    )
    assert resp.status_code == 400


def test_api_files_lists_repo_root():
    resp = client.get("/api/files", params={"path": ""})
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "dir"
    names = {e["name"] for e in body["entries"]}
    assert "src" in names
    # excluded dirs must never appear, even though they exist on disk
    assert ".git" not in names and ".venv" not in names


@pytest.mark.parametrize(
    "traversal",
    ["../../../etc/passwd", "..%2f..%2f..%2fetc%2fpasswd", "/etc/passwd", "checkpoints/../../etc/passwd"],
)
def test_api_files_blocks_path_traversal(traversal):
    resp = client.get("/api/files", params={"path": traversal})
    assert resp.status_code in (403, 404)


def test_api_files_excludes_venv_and_git():
    for excluded in [".venv", ".git", "data/Dataset"]:
        resp = client.get("/api/files", params={"path": excluded})
        assert resp.status_code == 404
