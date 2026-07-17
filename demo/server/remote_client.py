"""HTTP client for the remote optical inference server (demo/remote/optic_server.py).

Any failure — connect error, timeout, non-200 status, malformed JSON — raises
RemoteUnavailable so the caller (demo/server/app.py) can degrade the optical
path to the local fake engine without interrupting the demo.
"""
import os

import httpx

TIMEOUT_S = 30.0
DEFAULT_BASE_URL = "http://127.0.0.1:8765"


class RemoteUnavailable(Exception):
    """The remote optical server cannot serve this request."""


def _base_url(base_url=None):
    return (base_url or os.environ.get("OPTIC_REMOTE_URL")
            or DEFAULT_BASE_URL).rstrip("/")


def _call(method, path, base_url=None, **kwargs):
    url = _base_url(base_url) + path
    try:
        with httpx.Client(timeout=TIMEOUT_S) as client:
            resp = client.request(method, url, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except (httpx.HTTPError, ValueError) as e:
        # HTTPError: transport + non-2xx status; ValueError: bad JSON body
        raise RemoteUnavailable(f"{method} {url} failed: {e}") from e


def health(base_url=None):
    """GET /health → {"status", "engine", "weight", "uptime_s"}."""
    return _call("GET", "/health", base_url)


def infer(image_b64, base_url=None):
    """POST /infer with a jpeg/png b64 → PathResult dict (api.md)."""
    return _call("POST", "/infer", base_url, json={"image_b64": image_b64})
