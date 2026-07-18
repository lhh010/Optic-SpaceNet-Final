"""Tests for demo/server/compare.py (api.md comparison-fields contract)."""
import base64

import numpy as np
import torch

from conftest import REPO_ROOT  # noqa: F401  (sys.path bootstrap side effect)
from demo.server import compare
from demo.server.inference_local import encode_act_b64


def _t(a):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).unsqueeze(0)


def _layer(name, act, where="optical"):
    return {"name": name, "where": where, "spec": "s",
            "shape": list(act.shape), "latency_s": 0.0,
            "act_b64": encode_act_b64(_t(act))}


def test_mops_table_reconciles_with_official_metrics():
    total = sum(compare.LAYER_MOPS.values())
    assert abs(total - 1.0511) < 1e-3
    optical = sum(v for k, v in compare.LAYER_MOPS.items() if k != "stem")
    assert abs(optical / total - 0.9065) < 1e-3


def test_compare_acts_identical():
    a = np.random.default_rng(0).normal(size=(16, 16, 16)).astype(np.float32)
    r = compare.compare_acts(a, a)
    assert r["cos_sim"] == 1.0
    assert r["max_abs_err"] == 0.0
    assert sum(r["rel_err_hist"]["counts"]) == a.size
    assert r["rel_err_hist"]["counts"][0] == a.size   # rel err 全为 0 → 首桶


def test_compare_acts_known_values():
    b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    a = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    r = compare.compare_acts(a, b)
    assert r["cos_sim"] == 1.0                        # 共线向量
    assert r["max_abs_err"] == 0.5
    # rel: 0.5/1.001≈0.4995 → 桶 [0.25,0.5) idx5; 两个 0 → idx0
    assert r["rel_err_hist"]["counts"] == [2, 0, 0, 0, 0, 1, 0, 0, 0]


def test_compare_acts_zero_norms():
    z = np.zeros((4,), dtype=np.float32)
    assert compare.compare_acts(z, z)["cos_sim"] == 1.0
    assert compare.compare_acts(np.ones(4, np.float32), z)["cos_sim"] == 0.0


def test_inject_comparison_sets_fields_on_optical_layers():
    rng = np.random.default_rng(0)
    fp32 = {"layers": [_layer("stem", rng.normal(size=(8, 64, 64)), "electronic"),
                       _layer("fc2", rng.normal(size=(10,)))]}
    optical = {"layers": [_layer("stem", rng.normal(size=(8, 64, 64)), "electronic"),
                          _layer("fc2", rng.normal(size=(10,)))]}
    compare.inject_comparison(fp32, optical)
    for path in (fp32, optical):
        stem, fc2 = path["layers"]
        assert stem["cos_sim"] is None
        assert stem["max_abs_err"] is None
        assert stem["rel_err_hist"] is None
        assert stem["mops"] is None
        assert stem["theoretical_s"] is None
        assert -1.0 <= fc2["cos_sim"] <= 1.0
        assert fc2["max_abs_err"] >= 0
        assert sum(fc2["rel_err_hist"]["counts"]) == 10
        assert len(fc2["rel_err_hist"]["counts"]) == len(fc2["rel_err_hist"]["edges"]) + 1
        assert fc2["mops"] == compare.LAYER_MOPS["fc2"]
        assert fc2["theoretical_s"] == round(compare.LAYER_MOPS["fc2"] / 2.6, 6)


def test_inject_comparison_survives_undecodable_layer():
    """act 解码失败: 对比字段留 None, mops 静态值仍注入, 不炸响应。"""
    rng = np.random.default_rng(0)
    fp32 = {"layers": [_layer("fc2", rng.normal(size=(10,)))]}
    optical = {"layers": [_layer("fc2", rng.normal(size=(10,)))]}
    optical["layers"][0]["act_b64"] = base64.b64encode(b"garbage").decode()
    compare.inject_comparison(fp32, optical)
    assert optical["layers"][0]["cos_sim"] is None
    assert optical["layers"][0]["mops"] == compare.LAYER_MOPS["fc2"]
