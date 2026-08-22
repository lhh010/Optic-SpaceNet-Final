"""Tests for demo/server/render.py (api.md grid_b64 contract)."""
import base64
import io

import numpy as np
import torch
from PIL import Image

from conftest import REPO_ROOT  # noqa: F401  (sys.path bootstrap side effect)
from demo.server import render
from demo.server.inference_local import encode_act_b64


def test_shared_normalization_uses_joint_min_max():
    opt = np.full((4, 4), 3.0, dtype=np.float32)
    el = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    opt_u8, el_u8 = render._norm_pair(opt, el)
    assert opt_u8.min() == opt_u8.max() == 255      # joint max = 3.0
    assert set(np.unique(el_u8)) == {0, 85}         # 0→0, 1→85 (span 3)


def test_constant_activation_normalizes_to_zero():
    a = np.full((3, 3), 7.0, dtype=np.float32)
    a_u8, b_u8 = render._norm_pair(a, a)
    assert a_u8.max() == 0 and b_u8.max() == 0


def test_conv_grid_png_side_by_side():
    rng = np.random.default_rng(0)
    opt = rng.normal(size=(16, 16, 16)).astype(np.float32)
    el = rng.normal(size=(16, 16, 16)).astype(np.float32)
    png = render.render_layer_png("stage1", opt, el)
    img = Image.open(io.BytesIO(png))
    assert img.format == "PNG" and img.mode == "RGB"
    assert img.size == (2 * render.GRID + render.SEP, render.GRID)   # (246, 122)


def test_conv_grid_pads_when_fewer_than_16_channels():
    rng = np.random.default_rng(0)
    opt = rng.normal(size=(8, 64, 64)).astype(np.float32)   # stem: 8 channels
    el = rng.normal(size=(8, 64, 64)).astype(np.float32)
    img = Image.open(io.BytesIO(render.render_layer_png("stem", opt, el)))
    assert img.size == (2 * render.GRID + render.SEP, render.GRID)


def test_fc_strip_png_stacked():
    rng = np.random.default_rng(0)
    opt = rng.normal(size=(256,)).astype(np.float32)
    el = rng.normal(size=(256,)).astype(np.float32)
    img = Image.open(io.BytesIO(render.render_layer_png("fc1", opt, el)))
    assert img.size == (render.BAR_W, 2 * render.BAR_H + render.SEP)  # (256, 22)


def _layer(name, act):
    t = torch.from_numpy(np.ascontiguousarray(act, dtype=np.float32)).unsqueeze(0)
    return {"name": name, "where": "optical", "spec": "s",
            "shape": list(act.shape), "latency_s": 0.0,
            "act_b64": encode_act_b64(t)}


def test_inject_grids_sets_identical_grid_on_both_paths():
    rng = np.random.default_rng(0)
    fp32 = {"layers": [_layer("stage1", rng.normal(size=(16, 16, 16))),
                       _layer("fc2", rng.normal(size=(10,)))]}
    optical = {"layers": [_layer("stage1", rng.normal(size=(16, 16, 16))),
                          _layer("fc2", rng.normal(size=(10,)))]}
    render.inject_grids(fp32, optical)
    for f_layer, o_layer in zip(fp32["layers"], optical["layers"]):
        assert f_layer["grid_b64"] == o_layer["grid_b64"]
        img = Image.open(io.BytesIO(base64.b64decode(f_layer["grid_b64"])))
        assert img.format == "PNG"


def test_inject_grids_skips_undecodable_layer():
    """A corrupted act_b64 must not fail the whole response (live demo)."""
    rng = np.random.default_rng(0)
    fp32 = {"layers": [_layer("stage1", rng.normal(size=(16, 16, 16))),
                       _layer("fc2", rng.normal(size=(10,)))]}
    optical = {"layers": [_layer("stage1", rng.normal(size=(16, 16, 16))),
                          _layer("fc2", rng.normal(size=(10,)))]}
    optical["layers"][1]["act_b64"] = base64.b64encode(b"garbage").decode()
    render.inject_grids(fp32, optical)
    assert "grid_b64" in fp32["layers"][0]
    assert "grid_b64" in optical["layers"][0]
    assert "grid_b64" not in fp32["layers"][1]
    assert "grid_b64" not in optical["layers"][1]


def test_inject_grids_skips_layer_missing_on_fp32_side():
    """An optical layer with no fp32 counterpart is skipped, not KeyError."""
    rng = np.random.default_rng(0)
    fp32 = {"layers": [_layer("fc2", rng.normal(size=(10,)))]}
    optical = {"layers": [_layer("stage1", rng.normal(size=(16, 16, 16))),
                          _layer("fc2", rng.normal(size=(10,)))]}
    render.inject_grids(fp32, optical)
    assert "grid_b64" not in optical["layers"][0]
    assert "grid_b64" in optical["layers"][1]
    assert "grid_b64" in fp32["layers"][0]
