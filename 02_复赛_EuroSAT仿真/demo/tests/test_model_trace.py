"""Tests for demo/server/model_trace.py — segmented traced forward.

Covers (per demo/docs/design.md 测试策略):
  - 6 traced segments: names / where / spec / activation shapes;
  - segmented execution reproduces the whole-model logits (FP32, allclose);
  - per-layer latency_s >= 0;
  - FP32 model and optical-converted model (build_optical_model) expose the
    same hook points.
"""
import os

import pytest
import torch

from conftest import REPO_ROOT  # noqa: F401  (sys.path bootstrap side effect)
from demo.server import model_trace
from demo.server.model_trace import (
    EXPECTED_SHAPES,
    LAYER_SPECS,
    OpticSpaceNetStudent,
    build_student,
    forward_traced,
)

WEIGHT_PATH = os.path.join(REPO_ROOT, "weights", "spacenet_v2_phase4_v3_int8.pth")


@pytest.fixture(scope="module")
def model():
    return build_student(WEIGHT_PATH)


@pytest.fixture(scope="module")
def sample_input():
    torch.manual_seed(0)
    return torch.randn(1, 3, 64, 64)


def test_weight_loads_all_params():
    m = OpticSpaceNetStudent()
    info = model_trace.load_student_weights(m, WEIGHT_PATH)
    assert info["missing"] == [] and info["unexpected"] == []
    assert info["loaded"] == len(list(m.state_dict()))


def test_six_layers_meta(model, sample_input):
    out = forward_traced(model, sample_input)
    layers = out["layers"]
    assert [l["name"] for l in layers] == ["stem", "stage1", "stage2", "stage3", "fc1", "fc2"]
    assert [l["where"] for l in layers] == [
        "electronic", "optical", "optical", "optical", "optical", "optical"]
    for layer in layers:
        assert layer["spec"] == LAYER_SPECS[layer["name"]]
        assert tuple(layer["shape"]) == EXPECTED_SHAPES[layer["name"]], layer["name"]
        assert tuple(layer["act"].shape) == (1,) + EXPECTED_SHAPES[layer["name"]]


def test_segmented_logits_match_full_forward(model, sample_input):
    traced = forward_traced(model, sample_input)
    with torch.no_grad():
        ref = model(sample_input)
    assert torch.allclose(traced["logits"], ref, atol=1e-6)
    # fc2 activation is the logits
    assert torch.allclose(traced["layers"][-1]["act"], ref, atol=1e-6)


def test_layer_latencies_nonnegative(model, sample_input):
    out = forward_traced(model, sample_input)
    for layer in out["layers"]:
        assert layer["latency_s"] >= 0


def test_optical_model_hook_points_align(model, sample_input):
    from optic_layers import OpticalEngine, build_optical_model

    engine = OpticalEngine(use_real=False, verbose=False)
    optic = build_optical_model(model, engine, pad_to_8=True,
                                input_bit=8, weight_bit=8,
                                keep_first_conv_electronic=True)
    optic.eval()
    out = forward_traced(optic, sample_input)
    assert [l["name"] for l in out["layers"]] == ["stem", "stage1", "stage2", "stage3", "fc1", "fc2"]
    for layer in out["layers"]:
        assert tuple(layer["shape"]) == EXPECTED_SHAPES[layer["name"]]
        assert layer["latency_s"] >= 0
    assert out["logits"].shape == (1, 10)
