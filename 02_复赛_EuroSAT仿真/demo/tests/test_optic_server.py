"""Tests for demo/remote/optic_server.py — stdlib optical inference server.

Starts the real server (OPTIC_FAKE=1 semantics → fake engine) in a background
thread on an ephemeral port and drives it over HTTP with httpx: full
/health + /infer roundtrip against a real test-set jpeg, plus a 400 on a
corrupt image.  Nothing about the server under test is mocked.
"""
import base64
import threading

import httpx
import pytest

from conftest import REPO_ROOT  # noqa: F401  (sys.path bootstrap side effect)
import contract
import optic_server


def _test_jpeg_b64():
    import os

    from eurosat_split import split_indices
    from torchvision import datasets

    ds = datasets.ImageFolder(os.path.join(REPO_ROOT, "data", "EuroSAT_RGB"))
    _, _, test_idx = split_indices(len(ds), seed=42, val_ratio=0.2, test_ratio=0.2)
    path, _ = ds.samples[test_idx[0]]
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


@pytest.fixture(scope="module")
def server():
    srv = optic_server.create_server(host="127.0.0.1", port=0, fake=True)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def base_url(server):
    host, port = server.server_address[:2]
    return f"http://{host}:{port}"


def test_health(base_url):
    r = httpx.get(f"{base_url}/health", timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["engine"] == "fake-optical"
    assert body["weight"] == "spacenet_v2_phase4_v3_int8.pth"
    assert body["uptime_s"] >= 0


def test_infer_roundtrip(base_url):
    r = httpx.post(f"{base_url}/infer", json={"image_b64": _test_jpeg_b64()},
                   timeout=30)
    assert r.status_code == 200
    contract.check_path_result(r.json(), "fake-optical")


def test_infer_bad_image_400(base_url):
    bad = base64.b64encode(b"this is not an image").decode("ascii")
    r = httpx.post(f"{base_url}/infer", json={"image_b64": bad}, timeout=10)
    assert r.status_code == 400
    assert "error" in r.json()


def test_infer_malformed_body_400(base_url):
    r = httpx.post(f"{base_url}/infer", content=b"{not json", timeout=10)
    assert r.status_code == 400


def test_unknown_route_404(base_url):
    assert httpx.get(f"{base_url}/nope", timeout=10).status_code == 404


def test_import_has_no_side_effects():
    # importing the module must not bind a port or load a model
    assert not hasattr(optic_server, "_SERVER_STARTED")
