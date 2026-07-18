"""Server-side feature-map rendering for the optics demo.

Contract: demo/docs/api.md — each Layer gains ``grid_b64``: one PNG per layer
combining the optical and electronic activations, rendered here so the shared
min/max normalization stays server-side (the frontend only drops the PNG into
an <img>).

Rules (demo/docs/frontend-design.md):
  - per layer, joint min/max over concat(optical act, electronic act)
  - conv layers: first 16 channels as a 4×4 grid (channels beyond the available
    count stay dark), optical left / electronic right with a separator
  - fc layers: 1×N strips, optical top / electronic bottom
  - photon LUT: deep blue → cyan → bright white
"""
import base64
import io

import numpy as np
from PIL import Image

CELL = 28    # px per feature-map cell in a conv grid
PAD = 2      # px gap around/between cells
SEP = 2      # px separator between the optical/electronic halves
BAR_W = 256  # px width of an fc strip
BAR_H = 10   # px height of one fc strip
GRID = 4 * CELL + 5 * PAD          # 122, side of one path's 4×4 grid

_BG = np.array([15, 23, 42], dtype=np.uint8)   # slate-900 padding/separator

# photon LUT control points (x, r, g, b): deep blue → cyan → bright white
_LUT_X = [0, 128, 255]
_LUT_R = [10, 34, 255]
_LUT_G = [16, 211, 255]
_LUT_B = [48, 238, 255]


def decode_act(act_b64):
    """api.md: np.load(io.BytesIO(base64.b64decode(s)))["act"] → float16."""
    return np.load(io.BytesIO(base64.b64decode(act_b64)))["act"]


def _norm_pair(opt_act, el_act):
    """Joint min/max normalize both activations to uint8 (0..255)."""
    lo = float(min(opt_act.min(), el_act.min()))
    hi = float(max(opt_act.max(), el_act.max()))
    span = hi - lo
    if span <= 0:
        return (np.zeros_like(opt_act, dtype=np.uint8),
                np.zeros_like(el_act, dtype=np.uint8))

    def to_u8(a):
        return np.rint((a.astype(np.float32) - lo) / span * 255).astype(np.uint8)

    return to_u8(opt_act), to_u8(el_act)


def _lut(gray):
    """uint8 array (...) → photon-colored RGB uint8 (..., 3)."""
    x = np.arange(256)
    r = np.interp(x, _LUT_X, _LUT_R).astype(np.uint8)
    g = np.interp(x, _LUT_X, _LUT_G).astype(np.uint8)
    b = np.interp(x, _LUT_X, _LUT_B).astype(np.uint8)
    return np.stack([r[gray], g[gray], b[gray]], axis=-1)


def _grid_rgb(act_u8):
    """(C,H,W) uint8 → 4×4 cell grid RGB (GRID, GRID, 3); extra cells stay dark."""
    grid = np.tile(_BG, (GRID, GRID, 1))
    for c in range(min(16, act_u8.shape[0])):
        cell = Image.fromarray(act_u8[c]).resize((CELL, CELL), Image.NEAREST)
        row, col = divmod(c, 4)
        y = PAD + row * (CELL + PAD)
        x = PAD + col * (CELL + PAD)
        grid[y:y + CELL, x:x + CELL] = _lut(np.asarray(cell))
    return grid


def _strip_rgb(act_u8):
    """(N,) uint8 → BAR_H×BAR_W strip RGB."""
    row = act_u8.reshape(1, -1)
    img = Image.fromarray(row).resize((BAR_W, BAR_H), Image.NEAREST)
    return _lut(np.asarray(img))


def render_layer_png(name, opt_act, el_act):
    """One PNG (bytes) per layer: conv → side-by-side grids, fc → stacked strips."""
    opt_u8, el_u8 = _norm_pair(opt_act, el_act)
    if opt_act.ndim == 3:    # conv layer (C,H,W)
        left, right = _grid_rgb(opt_u8), _grid_rgb(el_u8)
        canvas = np.tile(_BG, (GRID, 2 * GRID + SEP, 1))
        canvas[:, :GRID] = left
        canvas[:, GRID + SEP:] = right
    else:                    # fc layer (N,)
        canvas = np.tile(_BG, (2 * BAR_H + SEP, BAR_W, 1))
        canvas[:BAR_H] = _strip_rgb(opt_u8)
        canvas[BAR_H + SEP:] = _strip_rgb(el_u8)
    buf = io.BytesIO()
    Image.fromarray(canvas).save(buf, format="PNG")
    return buf.getvalue()


def inject_grids(fp32, optical):
    """Pair layers by name, render each pair, set grid_b64 on both PathResults."""
    el_by_name = {layer["name"]: layer for layer in fp32["layers"]}
    for opt_layer in optical["layers"]:
        el_layer = el_by_name[opt_layer["name"]]
        png = render_layer_png(
            opt_layer["name"],
            decode_act(opt_layer["act_b64"]),
            decode_act(el_layer["act_b64"]))
        b64 = base64.b64encode(png).decode("ascii")
        opt_layer["grid_b64"] = b64
        el_layer["grid_b64"] = b64
