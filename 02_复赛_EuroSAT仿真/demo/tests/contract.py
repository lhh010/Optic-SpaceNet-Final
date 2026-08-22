"""Shared PathResult contract checker (demo/docs/api.md) for all demo tests."""
import base64
import io

import numpy as np

CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

LAYER_NAMES = ["stem", "stage1", "stage2", "stage3", "fc1", "fc2"]
LAYER_WHERE = ["electronic", "optical", "optical", "optical", "optical", "optical"]
LAYER_SHAPES = [[8, 64, 64], [16, 16, 16], [32, 8, 8], [16, 8, 8], [256], [10]]


def decode_act(act_b64):
    """api.md: np.load(io.BytesIO(base64.b64decode(s)))["act"] -> float16."""
    return np.load(io.BytesIO(base64.b64decode(act_b64)))["act"]


def check_path_result(res, engine):
    """Assert the full PathResult contract from demo/docs/api.md."""
    assert res["engine"] == engine
    assert res["pred"] in CLASSES

    probs = res["probs"]
    assert set(probs.keys()) == set(CLASSES), "probs must cover all 10 classes"
    assert abs(sum(probs.values()) - 1.0) < 1e-3, "softmax must sum to ~1"
    vals = list(probs.values())
    assert vals == sorted(vals, reverse=True), "probs must be sorted desc"
    assert probs and res["pred"] == max(probs, key=probs.get), "pred must be top-1"

    assert isinstance(res["latency_total_s"], (int, float))
    assert res["latency_total_s"] >= 0

    layers = res["layers"]
    assert [l["name"] for l in layers] == LAYER_NAMES
    assert [l["where"] for l in layers] == LAYER_WHERE
    for layer, shape in zip(layers, LAYER_SHAPES):
        assert isinstance(layer["spec"], str) and layer["spec"]
        assert layer["shape"] == shape, f"{layer['name']} shape"
        assert layer["latency_s"] >= 0
        act = decode_act(layer["act_b64"])
        assert act.dtype == np.float16
        assert list(act.shape) == layer["shape"]
    return res


def check_comparison_fields(path):
    """Fields injected by demo/server/compare.py at /api/infer aggregation."""
    for layer in path["layers"]:
        assert {"cos_sim", "max_abs_err", "rel_err_hist",
                "mops", "theoretical_s"} <= set(layer)
        if layer["where"] == "electronic":
            for k in ("cos_sim", "max_abs_err", "rel_err_hist",
                      "mops", "theoretical_s"):
                assert layer[k] is None, f"stem.{k} must be null"
            continue
        assert -1.0 <= layer["cos_sim"] <= 1.0
        assert layer["max_abs_err"] >= 0
        hist = layer["rel_err_hist"]
        n = 1
        for d in layer["shape"]:
            n *= d
        assert sum(hist["counts"]) == n, f"{layer['name']} hist total"
        assert len(hist["counts"]) == len(hist["edges"]) + 1
        assert all(c >= 0 for c in hist["counts"])
        assert layer["mops"] > 0
        assert layer["theoretical_s"] == round(layer["mops"] / 2.6, 6)
