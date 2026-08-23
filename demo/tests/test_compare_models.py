"""Routing contract for the Model 3 / M9 / M10 comparison page."""

import pytest

from conftest import REPO_ROOT  # noqa: F401
from demo.server import compare_models


def _result(engine):
    return {
        "engine": engine,
        "pred": "Forest",
        "probs": {"Forest": 1.0},
        "layers": [],
        "latency_total_s": 1.0,
    }


@pytest.mark.parametrize("model_id", [3, 9, 10])
@pytest.mark.parametrize(
    "backend", ["osimulator", "gazelle_ssh", "gazelle_serial"])
def test_every_model_uses_selected_backend(monkeypatch, model_id, backend):
    calls = []

    def selected_infer(image_b64, model_name=None, clean=False,
                       backend_mode=None):
        calls.append((model_name, clean, backend_mode))
        return _result("selected-engine")

    monkeypatch.setattr(compare_models.gazelle_client, "infer", selected_infer)
    out = compare_models.infer_model(model_id, "image", backend)

    assert calls == [("model%d" % model_id, False, backend)]
    assert out["backend"] == backend
    assert out["target"] == compare_models.BACKEND_TARGETS[backend]
    assert out["degraded"] is False


def test_selected_backend_failure_is_explicit_numpy_fallback(monkeypatch):
    calls = []

    def selected_infer(image_b64, model_name=None, clean=False,
                       backend_mode=None):
        calls.append((model_name, clean, backend_mode))
        if not clean:
            raise compare_models.GazelleUnavailable("board unreachable")
        return _result("numpy-clean")

    monkeypatch.setattr(compare_models.gazelle_client, "infer", selected_infer)
    out = compare_models.infer_model(10, "image", "gazelle_serial")

    assert calls == [
        ("model10", False, "gazelle_serial"),
        ("model10", True, None),
    ]
    assert out["engine"] == "numpy-clean"
    assert out["backend"] == "gazelle_serial"
    assert out["degraded"] is True
    assert "board unreachable" in out["degraded_reason"]


def test_unknown_comparison_model_rejected():
    with pytest.raises(ValueError, match="3, 9, or 10"):
        compare_models.infer_model(1, "image", "osimulator")


def test_unknown_comparison_backend_rejected():
    with pytest.raises(ValueError, match="backend must be"):
        compare_models.infer_model(3, "image", "unknown")
