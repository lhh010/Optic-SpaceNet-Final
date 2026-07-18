"""Tests for demo/server/app.py (FastAPI) + demo/server/metrics.py.

Full API surface via TestClient.  The degraded path is exercised against a
guaranteed-closed remote port; the happy remote path against a real
optic_server thread (fake engine).  /api/sample serves real test-set images.
"""
import base64
import io
import os
import socket
import threading

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from conftest import REPO_ROOT  # noqa: F401  (sys.path bootstrap side effect)
import contract
import optic_server
from demo.server.app import app
from demo.server.metrics import METRICS

CLASSES = contract.CLASSES


def _dead_url():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def remote_server():
    srv = optic_server.create_server(host="127.0.0.1", port=0, fake=True)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def sample_image(client):
    r = client.get("/api/sample")
    assert r.status_code == 200
    return r.json()


def test_health_local_and_remote_down(client, monkeypatch):
    monkeypatch.setenv("OPTIC_REMOTE_URL", _dead_url())
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["local"] == "ok"
    assert body["remote"] == "down"


def test_health_remote_up(client, monkeypatch, remote_server):
    host, port = remote_server.server_address[:2]
    monkeypatch.setenv("OPTIC_REMOTE_URL", f"http://{host}:{port}")
    body = client.get("/api/health").json()
    assert body == {"local": "ok", "remote": "fake-optical"}


def test_sample_contract(client):
    r = client.get("/api/sample")
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in CLASSES
    assert isinstance(body["index"], int) and body["index"] >= 0
    assert body["classes"] == CLASSES
    img = Image.open(io.BytesIO(base64.b64decode(body["image_b64"])))
    assert img.format == "JPEG" and img.size == (64, 64)


def test_sample_class_filter(client):
    r = client.get("/api/sample", params={"class": "Forest"})
    assert r.status_code == 200
    assert r.json()["label"] == "Forest"


def test_sample_unknown_class_404(client):
    assert client.get("/api/sample", params={"class": "Nope"}).status_code == 404


def test_metrics_exact(client):
    r = client.get("/api/metrics")
    assert r.status_code == 200
    assert r.json() == METRICS == {
        "optic_ratio": 0.9065, "mops_total": 1.0511, "mops_vs_model1": "150×",
        "osim_full_acc": 0.9028, "osim_full_n": 5400, "hw_align": 0.996,
        "val_int8": 0.9183, "params": 267944, "per_image_s": 2.5,
    }


def test_infer_via_remote(client, monkeypatch, remote_server, sample_image):
    host, port = remote_server.server_address[:2]
    monkeypatch.setenv("OPTIC_REMOTE_URL", f"http://{host}:{port}")
    r = client.post("/api/infer", json={
        "image_b64": sample_image["image_b64"], "label": sample_image["label"]})
    assert r.status_code == 200
    body = r.json()
    contract.check_path_result(body["fp32"], "fp32-local")
    contract.check_path_result(body["optical"], "fake-optical")
    meta = body["meta"]
    assert meta["degraded"] is False
    assert meta["label"] == sample_image["label"]
    assert isinstance(meta["correct"], bool)
    assert meta["remote_latency_s"] >= 0
    assert (body["optical"]["pred"] == meta["label"]) == meta["correct"]


def test_infer_degraded_when_remote_down(client, monkeypatch, sample_image):
    monkeypatch.setenv("OPTIC_REMOTE_URL", _dead_url())
    r = client.post("/api/infer", json={
        "image_b64": sample_image["image_b64"], "label": sample_image["label"]})
    assert r.status_code == 200
    body = r.json()
    assert body["meta"]["degraded"] is True
    assert body["optical"]["engine"] == "fake-optical"
    contract.check_path_result(body["fp32"], "fp32-local")
    contract.check_path_result(body["optical"], "fake-optical")


def test_infer_upload_without_label(client, monkeypatch, sample_image):
    monkeypatch.setenv("OPTIC_REMOTE_URL", _dead_url())
    r = client.post("/api/infer", json={"image_b64": sample_image["image_b64"]})
    assert r.status_code == 200
    meta = r.json()["meta"]
    assert meta["label"] is None and meta["correct"] is None


def test_infer_bad_image_400(client):
    bad = base64.b64encode(b"definitely not an image").decode("ascii")
    assert client.post("/api/infer", json={"image_b64": bad}).status_code == 400


def test_static_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "<html" in r.text.lower()


def test_infer_layers_carry_decodable_grids(client, monkeypatch, sample_image):
    monkeypatch.setenv("OPTIC_REMOTE_URL", _dead_url())
    r = client.post("/api/infer", json={
        "image_b64": sample_image["image_b64"], "label": sample_image["label"]})
    assert r.status_code == 200
    body = r.json()
    for path in (body["fp32"], body["optical"]):
        assert [l["name"] for l in path["layers"]] == contract.LAYER_NAMES
        for layer, shape in zip(path["layers"], contract.LAYER_SHAPES):
            img = Image.open(io.BytesIO(base64.b64decode(layer["grid_b64"])))
            assert img.format == "PNG"
            expected = (246, 122) if len(shape) == 3 else (256, 22)
            assert img.size == expected, layer["name"]
