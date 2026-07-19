"""Per-layer numerical comparison for the optics demo (scroll-narrative frontend).

Contract: demo/docs/api.md — each Layer gains comparison fields, computed here
at /api/infer aggregation time (same injection pattern as render.inject_grids):

  cos_sim        cosine similarity of flattened optical/fp32 activations
  max_abs_err    max |optical - fp32|
  rel_err_hist   {"edges": [...8 upper bounds...], "counts": [...9 ints...]}
                 counts[i] = elements with rel err in [edges[i-1], edges[i]),
                 edges[-1] := 0 implicitly, counts[8] = rel err >= edges[7];
                 rel err = |delta| / (|fp32| + REL_EPS)
  mops           static per-layer MOPs (official accounting, sums to 1.0511)
  theoretical_s  mops / CHIP_MOPS_PER_S (official 2.6 M int8 OP/s on-chip)

stem (electronic boundary) gets all fields None — no optical output to compare.
Simulator/engine wall-clock latency is meaningless for the optical layers and
is intentionally NOT surfaced here; the frontend shows theoretical_s instead.
"""
import numpy as np

from demo.server.render import decode_act

CHIP_MOPS_PER_S = 2.6   # official on-chip int8 throughput (M OP/s)
REL_EPS = 1e-3          # relative-error stabilizer
REL_EDGES = [1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]   # 8 bounds → 9 buckets

# Official MOPs split (sums to mops_total=1.0511 in demo/server/metrics.py;
# optical five layers sum to 0.9529 = 90.65%).  stage1's conv is counted at
# the post-MaxPool 16x16 resolution, matching the official accounting.
LAYER_MOPS = {
    "stem": 0.0983, "stage1": 0.1311, "stage2": 0.5243,
    "stage3": 0.0328, "fc1": 0.2621, "fc2": 0.0026,
}

_NULLS = {"cos_sim": None, "max_abs_err": None, "rel_err_hist": None,
          "mops": None, "theoretical_s": None}


def compare_acts(opt_act, el_act):
    """Two same-shape activations → {cos_sim, max_abs_err, rel_err_hist}."""
    a = opt_act.astype(np.float32).ravel()
    b = el_act.astype(np.float32).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 and nb == 0.0:
        cos = 1.0
    elif na == 0.0 or nb == 0.0:
        cos = 0.0
    else:
        cos = float(np.dot(a, b) / (na * nb))
    diff = np.abs(a - b)
    rel = diff / (np.abs(b) + REL_EPS)
    counts = np.histogram(rel, bins=[0.0] + REL_EDGES + [np.inf])[0]
    return {
        "cos_sim": round(cos, 6),
        "max_abs_err": round(float(diff.max()) if diff.size else 0.0, 6),
        "rel_err_hist": {"edges": list(REL_EDGES),
                         "counts": [int(c) for c in counts]},
    }


def inject_comparison(fp32, optical):
    """Pair layers by name; set comparison fields on both PathResults.

    stem gets explicit nulls.  Layers that fail to pair/decode keep nulls for
    the computed fields (mops/theoretical_s are static and still set) instead
    of failing the whole response — the live demo must stay displayable.
    """
    el_by_name = {layer["name"]: layer for layer in fp32["layers"]}
    for opt_layer in optical["layers"]:
        name = opt_layer["name"]
        el_layer = el_by_name.get(name)
        if el_layer is None:
            continue
        pair = (opt_layer, el_layer)
        for layer in pair:
            layer.update(_NULLS)
        if opt_layer.get("where") == "electronic":
            continue
        mops = LAYER_MOPS.get(name)
        for layer in pair:
            layer["mops"] = mops
            layer["theoretical_s"] = (
                round(mops / CHIP_MOPS_PER_S, 6) if mops is not None else None)
        try:
            cmp = compare_acts(decode_act(opt_layer["act_b64"]),
                               decode_act(el_layer["act_b64"]))
        except Exception as e:
            print(f"[compare] {name}: comparison failed: {e}")
            continue
        for layer in pair:
            layer.update(cmp)
