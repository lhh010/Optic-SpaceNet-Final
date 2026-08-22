"""Tests for demo/server/remote_client.py.

Happy path runs against a real optic_server (fake engine) in a thread; the
failure path points the client at a closed port and expects RemoteUnavailable.
"""
import base64
import os
import socket
import threading

import pytest

from conftest import REPO_ROOT  # noqa: F401  (sys.path bootstrap side effect)
import contract
import optic_server
from demo.server import remote_client
from demo.server.remote_client import RemoteUnavailable


def _test_jpeg_b64():
    from eurosat_split import split_indices
    from torchvision import datasets

    ds = datasets.ImageFolder(os.path.join(REPO_ROOT, "data", "EuroSAT_RGB"))
    _, _, test_idx = split_indices(len(ds), seed=42, val_ratio=0.2, test_ratio=0.2)
    path, _ = ds.samples[test_idx[0]]
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _dead_url():
    """http URL on a port that is guaranteed closed right now."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}"


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


def test_health_ok(base_url):
    body = remote_client.health(base_url)
    assert body["status"] == "ok"
    assert body["engine"] == "fake-optical"


def test_infer_ok(base_url):
    res = remote_client.infer(_test_jpeg_b64(), base_url)
    contract.check_path_result(res, "fake-optical")


def test_default_base_url_from_env(monkeypatch, base_url):
    monkeypatch.setenv("OPTIC_REMOTE_URL", base_url)
    assert remote_client.health()["status"] == "ok"


def test_health_dead_port_raises():
    with pytest.raises(RemoteUnavailable):
        remote_client.health(_dead_url())


def test_infer_dead_port_raises():
    with pytest.raises(RemoteUnavailable):
        remote_client.infer(_test_jpeg_b64(), _dead_url())


def test_http_error_status_raises(base_url):
    # server answers 400 for undecodable images → client surfaces
    # RemoteUnavailable so the caller degrades to the local fake engine
    bad = base64.b64encode(b"not an image").decode("ascii")
    with pytest.raises(RemoteUnavailable):
        remote_client.infer(bad, base_url)
