# 单栏滚动叙事版前端 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 demo 前端从单屏三栏 Dashboard 改为 7 屏单栏滚动叙事(scrollytelling), 并在后端逐层注入光|电数值对比字段。

**Architecture:** 选图即自动发起一次 `/api/infer`; 后端新增 `compare.py` 在聚合阶段(与 `render.inject_grids` 同一位置)就地注入 `cos_sim/max_abs_err/rel_err_hist/mops/theoretical_s`; 前端用原生 IntersectionObserver + CSS scroll-snap + CSS transition 做逐屏揭示动画, 零新依赖零构建。

**Tech Stack:** Python (numpy) + FastAPI; HTML + Tailwind Play CDN (vendored) + vanilla JS; pytest。

**Spec:** `demo/docs/frontend-scroll-design.md`(已确认)。契约: `demo/docs/api.md`。

## Global Constraints

- 零外网请求、零新前端依赖、零构建步骤; vendor/ 资产本地化。
- 测试一律从 repo root 运行: `cd Ltsimulator-test && python3 -m pytest demo/tests -q`。
- 保留现有视觉语言: ink/panel/edge/photon/gold/elec 色板, JetBrains Mono, 辉光/脉冲。
- 芯片官方吞吐: 2.6 M int8 OP/s; 光算层只展示理论上板耗时, 不展示模拟器/引擎实测耗时。
- stem 为电层: 对比字段全 null, 只展示结构+激活+实测耗时。
- commit 风格沿用仓库: 小写祈使句, 无 conventional 前缀 (如 `add demo page data flow: ...`)。
- git commit 需用户确认后执行。

---

### Task 1: compare.py — 逐层数值对比模块

**Files:**
- Create: `demo/server/compare.py`
- Test: `demo/tests/test_compare.py`

**Interfaces:**
- Consumes: `demo.server.render.decode_act(act_b64) -> np.ndarray` (已存在); `demo.server.inference_local.encode_act_b64` (测试用, 已存在)。
- Produces:
  - `compare.LAYER_MOPS: dict[str, float]` — 6 层静态 MOPs 表
  - `compare.CHIP_MOPS_PER_S = 2.6`
  - `compare.REL_EDGES: list[float]` — 8 个相对误差分桶上界
  - `compare.compare_acts(opt_act, el_act) -> {"cos_sim": float, "max_abs_err": float, "rel_err_hist": {"edges": list, "counts": list[int]}}`
  - `compare.inject_comparison(fp32, optical) -> None` — 就地注入, 两 PathResult 的每层增加 5 个键: `cos_sim, max_abs_err, rel_err_hist, mops, theoretical_s`; stem(where=="electronic")全 None; 失败层保留 None 不炸响应

MOPs 静态表口径(已与官方 `mops_total=1.0511` / `optic_ratio=0.9065` 对账: stage1 conv 按 MaxPool 后 16×16 分辨率计):
stem 0.0983 / stage1 0.1311 / stage2 0.5243 / stage3 0.0328 / fc1 0.2621 / fc2 0.0026。
合计 1.0512 ≈ 1.0511; 光算 5 层合计 0.9529, 占比 90.65%。

- [ ] **Step 1: Write the failing test**

创建 `demo/tests/test_compare.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Ltsimulator-test && python3 -m pytest demo/tests/test_compare.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'demo.server.compare'`

- [ ] **Step 3: Write minimal implementation**

创建 `demo/server/compare.py`:

```python
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
        except Exception:
            continue
        for layer in pair:
            layer.update(cmp)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd Ltsimulator-test && python3 -m pytest demo/tests/test_compare.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add demo/server/compare.py demo/tests/test_compare.py
git commit -m "add per-layer numerical comparison module for scroll demo"
```

---

### Task 2: 接入 /api/infer + 契约检查 + api.md 同步

**Files:**
- Modify: `demo/server/app.py:36` (import 行) 与 `demo/server/app.py:131` (inject 处)
- Modify: `demo/tests/contract.py` (追加 helper)
- Modify: `demo/tests/test_app.py` (追加测试)
- Modify: `demo/docs/api.md` (Layer 契约)

**Interfaces:**
- Consumes: `compare.inject_comparison(fp32, optical)` (Task 1)。
- Produces: `contract.check_comparison_fields(path) -> None` — 断言 5 个新键存在、stem 全 None、光算层 cos∈[-1,1]、hist counts 总数==shape 元素数、`theoretical_s == round(mops/2.6, 6)`。前端 Task 3/4 依赖响应中每层均有这 5 个键。

- [ ] **Step 1: Write the failing test**

`demo/tests/contract.py` 末尾追加:

```python
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
```

`demo/tests/test_app.py` 末尾追加:

```python
def test_infer_layers_carry_comparison_fields(client, monkeypatch, sample_image):
    monkeypatch.setenv("OPTIC_REMOTE_URL", _dead_url())
    r = client.post("/api/infer", json={
        "image_b64": sample_image["image_b64"], "label": sample_image["label"]})
    assert r.status_code == 200
    body = r.json()
    contract.check_comparison_fields(body["fp32"])
    contract.check_comparison_fields(body["optical"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Ltsimulator-test && python3 -m pytest demo/tests/test_app.py::test_infer_layers_carry_comparison_fields -q`
Expected: FAIL — `AssertionError` (键不存在)

- [ ] **Step 3: Wire compare into /api/infer**

`demo/server/app.py:36` 改为:

```python
from demo.server import compare, inference_local, remote_client, render  # noqa: E402
```

`demo/server/app.py:131` 之后(`render.inject_grids(fp32, optical)` 下一行)插入:

```python
    compare.inject_comparison(fp32, optical)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd Ltsimulator-test && python3 -m pytest demo/tests -q`
Expected: 全绿 (原 35 + 新增 7)

- [ ] **Step 5: Sync api.md**

`demo/docs/api.md` 的 `Layer` jsonc 块中 `"grid_b64"` 行之后追加:

```jsonc
  "cos_sim": 0.9987,           // 光|电激活展平余弦相似度; stem(电层)为 null
  "max_abs_err": 0.031,        // max |光-电|; stem 为 null
  "rel_err_hist": {            // 相对误差 |Δ|/(|fp32|+1e-3) 分桶; stem 为 null
    "edges": [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
    "counts": [/* 9 个 int: counts[i]=[edges[i-1],edges[i]) 内元素数,
                  edges[-1]:=0, counts[8]=≥2.0; 总和=激活元素数 */]
  },
  "mops": 0.5243,              // 静态表(compare.LAYER_MOPS), 与 mops_total/optic_ratio 对账; stem 为 null
  "theoretical_s": 0.2016,     // mops / 2.6 (芯片官方 2.6 M int8 OP/s 的上板估算); stem 为 null
```

并在渲染规则段落后追加一行:

```markdown
- 对比字段 (`cos_sim` 等 5 个) 由本地后端 `demo/server/compare.py` 在 `/api/infer`
  聚合阶段就地注入 (与 `grid_b64` 同一位置); 远程服务不返回这些字段。
  光算层的模拟器/引擎实测耗时无物理意义, 前端展示 `theoretical_s` 而非 `latency_s`。
```

- [ ] **Step 6: Commit**

```bash
git add demo/server/app.py demo/tests/contract.py demo/tests/test_app.py demo/docs/api.md
git commit -m "inject per-layer comparison fields into /api/infer response"
```

---

### Task 3: index.html — 7 屏滚动结构 + 动画 CSS

**Files:**
- Modify: `demo/web/index.html` (整体重写)

**Interfaces:**
- Consumes: 无 (纯静态结构)。
- Produces (Task 4 的 app.js 依赖这些 id/class):
  - 固定元素: `#progress`(顶部进度条), `#health-dot`, `#health-text`, `#dots`(圆点导航容器), `#banner`(固定顶部警示条)
  - 屏 1 `#sec-input.screen`: `#chips`, `#img`, `#label`, `#index`, `#class-select`, `#btn-sample`, `#file`, `#infer-status`, `#stem-body`
  - `#stages` 容器: app.js 生成 5 个 `<section class="screen" data-stage="..." data-label="...">`, 内部 class 约定: `.stage-title/.stage-spec/.stage-fig/.stage-cos/.stage-maxabs b/.stage-hist/.stage-theo/.stage-mops/.stage-elec`
  - 屏 7 `#sec-result.screen`: `#pred-fp32`, `#pred-opt`, `#verdict`, `#probs`, `#metrics`
  - CSS 契约: `.screen`(100vh+snap), `.reveal-item`/`.revealed`(揭示动画), `.skeleton`(骨架脉冲), `.hist-bar`(生长), `.dot`/`.dot.active`(导航), `.scroll-hint`(呼吸箭头)

不再有 `#btn-infer`/`#stopwatch`(选图即自动推理); 不再有中栏 `#layers`。

- [ ] **Step 1: Rewrite index.html**

完整替换 `demo/web/index.html`:

```html
<!DOCTYPE html>
<!-- Model 3 光计算演示 — 单栏滚动叙事版 (demo/docs/frontend-scroll-design.md)。
     零外网依赖: Tailwind Play CDN 与字体均已 vendored 到 vendor/。 -->
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>光计算推理演示 · Model 3</title>
<link rel="stylesheet" href="vendor/fonts.css">
<script src="vendor/tailwind.play.js"></script>
<script>
tailwind.config = {
  theme: { extend: {
    colors: { ink: "#05070f", panel: "#0b1120", edge: "#1e293b",
              photon: "#22d3ee", gold: "#fbbf24", elec: "#64748b" },
    fontFamily: { mono: ['"JetBrains Mono"', "ui-monospace",
                         "SFMono-Regular", "Menlo", "monospace"] },
  } },
};
</script>
<style type="text/tailwindcss">
  .card { @apply bg-panel border border-edge rounded-xl p-4; }
  .btn  { @apply rounded-lg px-3 py-2 text-sm font-bold transition cursor-pointer; }
  .chip { @apply border border-edge rounded-full px-3 py-1 text-xs text-slate-300 bg-panel; }
</style>
<style>
  html { scroll-snap-type: y proximity; scroll-behavior: smooth; }
  .screen { min-height: 100vh; scroll-snap-align: start;
            display: flex; align-items: center; justify-content: center; }

  .glow-photon { text-shadow: 0 0 12px rgba(34, 211, 238, .55); }
  .grid-img { image-rendering: pixelated; }
  .bar-photon { background: linear-gradient(90deg, #0891b2, #22d3ee);
                box-shadow: 0 0 8px rgba(34, 211, 238, .45); }
  .bar-elec { background: #475569; }

  /* 揭示动画: IO 命中且数据就绪后 .revealed 触发, 子项错峰上浮 */
  .reveal-item { opacity: 0; transform: translateY(24px);
                 transition: opacity .6s ease, transform .6s ease; }
  .revealed .reveal-item { opacity: 1; transform: none; }
  .revealed .reveal-item:nth-child(2) { transition-delay: .12s; }
  .revealed .reveal-item:nth-child(3) { transition-delay: .24s; }
  .revealed .reveal-item:nth-child(4) { transition-delay: .36s; }

  /* 骨架脉冲(推理结果未就绪) */
  @keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(34, 211, 238, .5); }
    50%      { box-shadow: 0 0 18px 4px rgba(34, 211, 238, .35); }
  }
  .skeleton { animation: pulse-glow 1.2s ease-in-out infinite; }
  .revealed .skeleton { animation: none; }

  /* 直方图 bar 生长(高度由 app.js 行内设置) */
  .hist-bar { height: 0; transition: height .7s cubic-bezier(.2,.8,.2,1); }

  /* 向下箭头呼吸 */
  @keyframes breathe {
    0%, 100% { transform: translateY(0); opacity: .5; }
    50%      { transform: translateY(8px); opacity: 1; }
  }
  .scroll-hint { animation: breathe 1.8s ease-in-out infinite; }

  /* 圆点导航 */
  .dot { width: 10px; height: 10px; border-radius: 9999px;
         background: #334155; transition: all .3s; cursor: pointer; }
  .dot.active { background: #22d3ee;
                box-shadow: 0 0 10px rgba(34, 211, 238, .8);
                transform: scale(1.35); }

  @media (prefers-reduced-motion: reduce) {
    html { scroll-behavior: auto; }
    .reveal-item, .hist-bar { transition: none !important; }
    .skeleton, .scroll-hint { animation: none !important; }
  }
</style>
</head>
<body class="bg-ink text-slate-200 font-mono min-w-[1280px]">

<div id="progress" class="fixed top-0 left-0 h-[2px] bg-photon z-50"
     style="width:0%; box-shadow:0 0 8px rgba(34,211,238,.6)"></div>

<header class="fixed top-[2px] left-0 right-0 z-40 px-5 py-2 flex items-center gap-3
               bg-ink/85 backdrop-blur border-b border-edge/50">
  <span class="text-sm font-bold text-photon glow-photon">⚡ 光计算推理演示</span>
  <span class="text-xs text-slate-500">Model 3 · SpaceNet V2 + KD · int8</span>
  <div class="flex items-center gap-2 ml-auto">
    <span id="health-dot" class="w-2.5 h-2.5 rounded-full bg-slate-600"></span>
    <span id="health-text" class="text-xs text-slate-400">连接中…</span>
  </div>
</header>

<nav id="dots" class="fixed right-4 top-1/2 -translate-y-1/2 z-40
                     flex flex-col gap-3"></nav>

<div id="banner" class="hidden fixed top-12 left-1/2 -translate-x-1/2 z-40"></div>

<!-- 屏 1: 首屏 — Header + 输入 + stem(电边界预处理) -->
<section id="sec-input" class="screen" data-label="输入">
  <div class="w-full max-w-[1100px] px-8 pt-14">
    <div class="text-center mb-6">
      <h1 class="text-3xl font-bold tracking-wide">
        <span class="text-photon glow-photon">⚡ 光计算推理演示</span>
      </h1>
      <div class="text-slate-400 mt-1">Model 3 · SpaceNet V2 + KD · int8 · EuroSAT 10 类</div>
      <div id="chips" class="flex gap-2 mt-4 justify-center flex-wrap"></div>
    </div>
    <div class="grid grid-cols-[320px_1fr] gap-6 items-start">
      <div class="card flex flex-col gap-3">
        <img id="img" alt="输入图像"
             class="w-full aspect-square rounded-lg border border-edge object-cover bg-black">
        <div class="text-xs leading-5">
          <div>label: <b id="label" class="text-gold">-</b></div>
          <div>index: <span id="index" class="text-slate-400">-</span></div>
        </div>
        <div class="flex gap-2">
          <select id="class-select"
                  class="bg-panel border border-edge rounded-lg text-xs px-2 py-2 flex-1"></select>
          <button id="btn-sample" class="btn bg-edge hover:bg-slate-700">抽图</button>
        </div>
        <label class="btn bg-edge hover:bg-slate-700 text-center text-xs">
          上传图片<input type="file" id="file" accept="image/*" class="hidden">
        </label>
        <div id="infer-status" class="text-center text-photon text-sm min-h-[20px]">抽图后自动开始推理</div>
      </div>
      <div class="card">
        <h2 class="text-sm text-slate-400 mb-2">
          stem <span class="text-[10px] px-1.5 py-0.5 rounded border border-elec/60 text-elec">电</span>
          <span class="text-xs text-slate-500 ml-2">电计算边界预处理 — 进入光域之前</span>
        </h2>
        <div id="stem-body" class="text-slate-500 text-sm">等待推理…</div>
      </div>
    </div>
    <div class="text-center mt-6">
      <div class="scroll-hint text-photon text-2xl">▼</div>
      <div class="text-xs text-slate-500 mt-1">滚动查看逐 stage 光|电对比</div>
    </div>
  </div>
</section>

<!-- 屏 2-6: stage 屏由 app.js 生成 -->
<div id="stages"></div>

<!-- 屏 7: 结果 + takeaway -->
<section id="sec-result" class="screen" data-label="结果">
  <div class="w-full max-w-[1100px] px-8">
    <h2 class="text-2xl font-bold text-center mb-6">结果 <span class="text-photon">·</span> 结论</h2>
    <div class="grid grid-cols-2 gap-6 items-start">
      <div class="card reveal-item">
        <h3 class="text-sm text-slate-400 mb-3">预测对比</h3>
        <div class="grid grid-cols-2 gap-3 text-center">
          <div>
            <div class="text-xs text-elec mb-1">FP32 电计算</div>
            <div id="pred-fp32" class="text-xl font-bold">-</div>
          </div>
          <div>
            <div class="text-xs text-photon mb-1">int8 光计算</div>
            <div id="pred-opt" class="text-xl font-bold text-photon glow-photon">-</div>
          </div>
        </div>
        <div id="verdict" class="text-center text-sm mt-2 min-h-[20px]"></div>
        <div id="probs" class="mt-3 flex flex-col gap-1.5"></div>
      </div>
      <div class="card reveal-item">
        <h3 class="text-sm text-slate-400 mb-2">关键指标</h3>
        <div id="metrics" class="text-xs leading-6"></div>
      </div>
    </div>
    <div class="card mt-6 text-center reveal-item">
      <div class="text-xs text-gold mb-2">TAKEAWAY</div>
      <p class="text-base leading-7">
        <b class="text-photon">90.65%</b> 算力光化, 总算力降 <b class="text-photon">150×</b>;
        int8 光计算 vs FP32 逐层高度一致, 全量 5400 张 osim 精度 <b class="text-photon">90.28%</b>
        —— 光计算在这颗模型上可行。
      </p>
    </div>
  </div>
</section>

<script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Verify served**

```bash
cd Ltsimulator-test && python3 -m uvicorn demo.server.app:app --port 8000 &
sleep 3 && curl -s http://localhost:8000/ | grep -c 'sec-input\|sec-result\|id="dots"\|id="progress"'
```

Expected: 输出 ≥ 4 处匹配; 页面此时 JS 报错不影响本步(app.js 在 Task 4 重写)。kill uvicorn 或留到 Task 4。

- [ ] **Step 3: Commit**

```bash
git add demo/web/index.html
git commit -m "rebuild demo page as 7-screen scroll narrative structure"
```

---

### Task 4: app.js — 滚动揭示 + 自动推理 + 数值动画

**Files:**
- Modify: `demo/web/app.js` (整体重写)

**Interfaces:**
- Consumes: Task 3 的 DOM 契约; `/api/infer` 响应中 optical/fp32 两 PathResult 的 Layer 含 `grid_b64`(拼接 PNG)、`cos_sim/max_abs_err/rel_err_hist/mops/theoretical_s`(Task 2 注入); `layers[0]` 为 stem。
- Produces: 无(叶子任务)。

行为要点(spec 交互节): 选图即自动推理; IO threshold 0.5 触发揭示; 数据未就绪显示骨架, 到达后自动揭示; 重新抽图/上传 → 全部重置再推理; 上传图不显示对错; `prefers-reduced-motion` 降级。URL 参数 `?autoinfer` 废弃(自动推理已是默认行为)。

- [ ] **Step 1: Rewrite app.js**

完整替换 `demo/web/app.js`:

```javascript
"use strict";
/* Model 3 光计算演示 — 单栏滚动叙事版 (demo/docs/frontend-scroll-design.md)。
   数据契约: demo/docs/api.md (Layer 含 grid_b64 + compare.py 注入的对比字段)。
   交互: 选图即自动推理一次; 滚动到某屏且数据就绪 → 揭示该屏动画。 */

const $ = (id) => document.getElementById(id);
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;

const STAGES = [
  { name: "stage1", label: "Stage 1" },
  { name: "stage2", label: "Stage 2" },
  { name: "stage3", label: "Stage 3" },
  { name: "fc1",    label: "FC 1" },
  { name: "fc2",    label: "FC 2 · logits" },
];

let current = null;        // {image_b64, label|null}
let inferData = null;      // /api/infer 响应体
let inferState = "idle";   // idle | running | done | error
let timer = null;
const revealed = new Set();   // 已揭示 section id

/* ---------- 基础 ---------- */

async function j(r) {
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`);
  return r.json();
}

function showBanner(kind, text) {
  const styles = {
    warn: "border-gold/60 text-gold bg-yellow-950/40",
    error: "border-red-500/60 text-red-400 bg-red-950/40",
  };
  const b = $("banner");
  b.className = `fixed top-12 left-1/2 -translate-x-1/2 z-40 rounded-lg border px-4 py-2 text-sm ${styles[kind]}`;
  b.textContent = text;
}
function hideBanner() { $("banner").className = "hidden"; }

// 推理期间锁定输入控件, 防止在途响应把旧图的结果渲染到新图上 (陈旧结果竞态)。
function lockInputs(lock) {
  $("btn-sample").disabled = lock;
  $("class-select").disabled = lock;
  $("file").disabled = lock;
  $("file").parentElement.classList.toggle("opacity-40", lock);
  $("file").parentElement.classList.toggle("pointer-events-none", lock);
}

function fmtLat(s) { return s >= 0.1 ? `${s.toFixed(2)}s` : `${(s * 1000).toFixed(1)}ms`; }

/* ---------- health & metrics (顶栏 / 屏1 chips / 屏7 指标) ---------- */

async function refreshHealth() {
  try {
    const h = await j(await fetch("/api/health"));
    const conf = {
      "gazelle-osim": ["bg-photon", "text-photon", "真机光计算 gazelle-osim 已连接"],
      "fake-optical": ["bg-gold", "text-gold", "降级模式: 本地 fake 引擎"],
      "down": ["bg-red-500", "text-red-400", "远程不可用 — 推理将走降级链"],
    }[h.remote] || ["bg-slate-600", "text-slate-400", `remote: ${h.remote}`];
    $("health-dot").className = `w-2.5 h-2.5 rounded-full ${conf[0]}`;
    $("health-text").className = `text-xs ${conf[1]}`;
    $("health-text").textContent = conf[2];
  } catch (e) {
    $("health-text").textContent = "health 检查失败";
  }
}

async function loadMetrics() {
  const m = await j(await fetch("/api/metrics"));
  const pct = (x, d = 2) => `${(x * 100).toFixed(d)}%`;
  const rows = [
    ["光计算占比", pct(m.optic_ratio)],
    ["总算力", `${m.mops_total} MOPs (vs Model 1: ${m.mops_vs_model1})`],
    ["osim 全量精度", `${pct(m.osim_full_acc)} (n=${m.osim_full_n})`],
    ["int8 val 精度", pct(m.val_int8)],
    ["硬件对齐率", pct(m.hw_align, 1)],
    ["参数量", m.params.toLocaleString()],
    ["理论上板耗时", `~${(m.mops_total * m.optic_ratio / 2.6).toFixed(2)}s @2.6M OP/s`],
  ];
  $("metrics").innerHTML = rows.map(([k, v]) =>
    `<div class="flex justify-between border-b border-edge/60 py-0.5">
       <span class="text-slate-400">${k}</span><b class="text-slate-100">${v}</b>
     </div>`).join("");
  $("chips").innerHTML = [
    ["光算占比", pct(m.optic_ratio)],
    ["算力削减", m.mops_vs_model1],
    ["osim 全量", `${pct(m.osim_full_acc)} (n=${m.osim_full_n})`],
    ["硬件对齐", pct(m.hw_align, 1)],
    ["int8 val", pct(m.val_int8)],
  ].map(([k, v]) => `<span class="chip">${k} <b class="text-photon">${v}</b></span>`).join("");
}

/* ---------- stage 屏 DOM 生成 ---------- */

function buildStages() {
  $("stages").innerHTML = STAGES.map((s) => `
  <section id="sec-${s.name}" class="screen" data-stage="${s.name}" data-label="${s.label}">
    <div class="w-full max-w-[1100px] px-8">
      <div class="text-center mb-5 reveal-item">
        <span class="stage-title text-2xl font-bold"></span>
        <span class="text-[10px] px-1.5 py-0.5 rounded border border-photon/50 text-photon ml-2">光</span>
        <div class="stage-spec text-xs text-slate-500 mt-2">&nbsp;</div>
      </div>
      <div class="grid grid-cols-[1fr_300px] gap-6 items-center">
        <div class="card reveal-item flex items-center justify-center min-h-[280px] skeleton stage-fig">
          <span class="text-slate-500 text-sm">光计算结果就绪后揭示 feature map…</span>
        </div>
        <div class="flex flex-col gap-4">
          <div class="card reveal-item">
            <div class="text-xs text-slate-400">余弦相似度 光|电</div>
            <div class="stage-cos text-3xl font-bold text-photon glow-photon">-</div>
            <div class="stage-maxabs text-xs text-slate-400 mt-2">max |Δ|: <b class="text-slate-100">-</b></div>
          </div>
          <div class="card reveal-item">
            <div class="text-xs text-slate-400 mb-2">相对误差分布 |Δ|/(|fp32|+ε)</div>
            <div class="stage-hist flex items-end gap-1 h-20"></div>
            <div class="flex justify-between text-[9px] text-slate-500 mt-1">
              <span>0</span><span>0.1</span><span>≥2</span>
            </div>
          </div>
          <div class="card reveal-item text-xs leading-6">
            <div>理论上板耗时 <b class="stage-theo text-photon">-</b></div>
            <div class="text-slate-500">@ 2.6 M int8 OP/s · <span class="stage-mops"></span></div>
            <div>电计算实测 <b class="stage-elec text-slate-200">-</b></div>
          </div>
        </div>
      </div>
    </div>
  </section>`).join("");
}

/* ---------- 圆点导航 + 进度条 ---------- */

function buildDots() {
  const secs = [...document.querySelectorAll(".screen")];
  $("dots").innerHTML = secs.map((s) =>
    `<div class="dot" data-target="${s.id}" title="${s.dataset.label || ""}"></div>`).join("");
  $("dots").querySelectorAll(".dot").forEach((d) => {
    d.onclick = () => document.getElementById(d.dataset.target)
      .scrollIntoView({ behavior: REDUCED ? "auto" : "smooth" });
  });
}

function activateDot(id) {
  $("dots").querySelectorAll(".dot").forEach((d) =>
    d.classList.toggle("active", d.dataset.target === id));
}

addEventListener("scroll", () => {
  const h = document.documentElement;
  const p = h.scrollTop / Math.max(1, h.scrollHeight - h.clientHeight);
  $("progress").style.width = `${(p * 100).toFixed(1)}%`;
}, { passive: true });

/* ---------- 揭示 ---------- */

function countUp(el, target, digits) {
  if (REDUCED) { el.textContent = target.toFixed(digits); return; }
  const t0 = performance.now(), dur = 800;
  (function tick(t) {
    const k = Math.min(1, (t - t0) / dur);
    el.textContent = (target * (1 - Math.pow(1 - k, 3))).toFixed(digits);
    if (k < 1) requestAnimationFrame(tick);
  })(t0);
}

function bucketLabel(edges, i) {
  if (i === 0) return `< ${edges[0]}`;
  if (i === edges.length) return `≥ ${edges[edges.length - 1]}`;
  return `${edges[i - 1]} ~ ${edges[i]}`;
}

function fillStage(sec, name) {
  const ol = inferData.optical.layers.find((l) => l.name === name);
  const fl = inferData.fp32.layers.find((l) => l.name === name);
  sec.querySelector(".stage-title").textContent = name;
  sec.querySelector(".stage-spec").textContent =
    `${ol.spec} · shape [${ol.shape.join(",")}]`;
  sec.querySelector(".stage-fig").innerHTML =
    `<img class="grid-img rounded border border-edge w-full" alt="${name}"
          src="data:image/png;base64,${ol.grid_b64}">`;
  countUp(sec.querySelector(".stage-cos"), ol.cos_sim, 4);
  sec.querySelector(".stage-maxabs b").textContent = ol.max_abs_err.toFixed(4);
  const hist = sec.querySelector(".stage-hist");
  const { edges, counts } = ol.rel_err_hist;
  const total = counts.reduce((a, b) => a + b, 0) || 1;
  const max = Math.max(...counts, 1);
  hist.innerHTML = counts.map((c, i) =>
    `<div class="hist-bar bar-photon flex-1 rounded-t"
          data-h="${c ? Math.max(3, (c / max) * 100) : 0}"
          title="rel err ${bucketLabel(edges, i)}: ${(c / total * 100).toFixed(1)}%"></div>`).join("");
  requestAnimationFrame(() => requestAnimationFrame(() =>
    hist.querySelectorAll(".hist-bar").forEach((b) =>
      b.style.height = `${b.dataset.h}%`)));
  sec.querySelector(".stage-theo").textContent = fmtLat(ol.theoretical_s);
  sec.querySelector(".stage-mops").textContent = `${ol.mops} MOPs`;
  sec.querySelector(".stage-elec").textContent = fmtLat(fl.latency_s);
}

function fillResult() {
  const { fp32, optical, meta } = inferData;
  refreshHealth();
  $("pred-fp32").textContent = fp32.pred;
  $("pred-opt").textContent = optical.pred;
  $("verdict").innerHTML = meta.correct === null
    ? `<span class="text-slate-500">上传图: 仅展示预测</span>`
    : meta.correct
      ? `<span class="text-photon">✓ 光路径预测正确 (label: ${meta.label})</span>`
      : `<span class="text-red-400">✗ 光路径预测错误 (label: ${meta.label})</span>`;
  const maxP = Math.max(...Object.values(optical.probs));
  $("probs").innerHTML = Object.entries(optical.probs).map(([cls, p]) => {
    const f = fp32.probs[cls] ?? 0;
    return `<div class="grid grid-cols-[110px_1fr_52px] items-center gap-2 text-[11px]">
      <span class="text-slate-400 truncate">${cls}</span>
      <div class="flex flex-col gap-0.5">
        <div class="bar-photon h-[7px] rounded" style="width:${(p / maxP) * 100}%"></div>
        <div class="bar-elec h-[5px] rounded" style="width:${(f / maxP) * 100}%"></div>
      </div>
      <span class="text-right text-photon">${(p * 100).toFixed(1)}</span>
    </div>`;
  }).join("");
}

function fillStem() {
  const ol = inferData.optical.layers[0];   // stem (电层, grid 为 光路|电路 两实现对比)
  const fl = inferData.fp32.layers[0];
  $("stem-body").innerHTML = `
    <div class="text-xs text-slate-500 mb-2">${ol.spec} · [${ol.shape.join(",")}]
      · 电实测 ${fmtLat(fl.latency_s)}</div>
    <img class="grid-img rounded border border-edge w-full" alt="stem"
         src="data:image/png;base64,${ol.grid_b64}">`;
}

function revealSection(sec) {
  if (revealed.has(sec.id)) return;
  const needsData = sec.dataset.stage || sec.id === "sec-result";
  if (needsData && inferState !== "done") return;   // 骨架等待, 数据到达后补揭示
  revealed.add(sec.id);
  sec.classList.add("revealed");
  if (sec.dataset.stage) fillStage(sec, sec.dataset.stage);
  if (sec.id === "sec-result") fillResult();
}

const io = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (e.intersectionRatio >= 0.5) {
      activateDot(e.target.id);
      revealSection(e.target);
    }
  }
}, { threshold: [0.5] });

function observeScreens() {
  document.querySelectorAll(".screen").forEach((s) => io.observe(s));
}

// 数据到达时补揭示「已在视口但因等数据而跳过」的屏
function revealVisible() {
  document.querySelectorAll(".screen").forEach((s) => {
    const r = s.getBoundingClientRect();
    if (r.top < innerHeight * 0.5 && r.bottom > innerHeight * 0.5) revealSection(s);
  });
}

function resetReveal() {
  revealed.clear();
  document.querySelectorAll(".screen.revealed")
    .forEach((s) => s.classList.remove("revealed"));
  buildStages();                 // stage 屏回骨架态
  observeScreens();              // 重建的节点需重新挂 IO (重复 observe 安全)
  buildDots();
  $("stem-body").textContent = "等待推理…";
}

/* ---------- 推理 (选图即自动) ---------- */

async function startInfer() {
  if (!current) return;
  resetReveal();
  lockInputs(true);
  inferState = "running";
  const t0 = performance.now();
  timer = setInterval(() => {
    $("infer-status").textContent =
      `光计算推理中… ${((performance.now() - t0) / 1000).toFixed(1)}s`;
  }, 100);
  try {
    const body = await j(await fetch("/api/infer", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(current),
    }));
    inferData = body;
    inferState = "done";
    hideBanner();
    if (body.meta.degraded) {
      showBanner("warn", "⚠ 远程光计算不可用, 已降级为本地 fake 引擎 (meta.degraded=true)");
    }
    fillStem();
    revealVisible();
    $("infer-status").textContent = "✓ 推理完成 · 滚动查看逐 stage 对比";
  } catch (e) {
    inferState = "error";
    showBanner("error", `推理失败: ${e.message}`);
    $("infer-status").textContent = "";
  } finally {
    clearInterval(timer);
    lockInputs(false);
  }
}

/* ---------- 输入 ---------- */

function setImage(b64, label, index) {
  current = { image_b64: b64, label: label ?? null };
  $("img").src = "data:image/jpeg;base64," + b64;
  $("label").textContent = label ?? "(上传图, 无 ground truth)";
  $("index").textContent = index ?? "-";
  startInfer();
}

async function loadSample(cls) {
  const url = cls && cls !== "random" ? `/api/sample?class=${cls}` : "/api/sample";
  const s = await j(await fetch(url));
  if ($("class-select").options.length === 0) {
    $("class-select").innerHTML =
      `<option value="random">随机类别</option>` +
      s.classes.map((c) => `<option value="${c}">${c}</option>`).join("");
  }
  setImage(s.image_b64, s.label, s.index);
}

$("btn-sample").onclick = () =>
  loadSample($("class-select").value).catch((e) => showBanner("error", e.message));

$("file").onchange = (ev) => {
  const f = ev.target.files[0];
  if (!f) return;
  const rd = new FileReader();
  rd.onload = () => { setImage(rd.result.split(",")[1], null, null); hideBanner(); };
  rd.readAsDataURL(f);
};

/* ---------- 启动 ---------- */

buildStages();
buildDots();
observeScreens();
refreshHealth();
loadMetrics().catch(() => {});
loadSample().catch((e) => showBanner("error", `抽图失败: ${e.message}`));
```

- [ ] **Step 2: 起服务 + 契约 smoke (降级模式)**

```bash
cd Ltsimulator-test && OPTIC_REMOTE_URL=http://127.0.0.1:1 python3 -m uvicorn demo.server.app:app --port 8000 &
sleep 3
IMG=$(curl -s http://localhost:8000/api/sample | python3 -c 'import sys,json;print(json.load(sys.stdin)["image_b64"])')
curl -s -X POST http://localhost:8000/api/infer -H 'Content-Type: application/json' \
  -d "{\"image_b64\": \"$IMG\", \"label\": \"Forest\"}" \
| python3 -c '
import sys, json
b = json.load(sys.stdin)
for l in b["optical"]["layers"]:
    print(l["name"], "cos=", l["cos_sim"], "theo=", l["theoretical_s"],
          "hist_sum=", None if l["rel_err_hist"] is None else sum(l["rel_err_hist"]["counts"]))
'
```

Expected: stem 行三个值均为 None; 其余 5 层 cos 为 [-1,1] 小数、theo 为数值、hist_sum 等于该层元素数 (4096/2048/1024/256/10)。

- [ ] **Step 3: 浏览器人工核对 (检查点)**

打开 `http://localhost:8000`, 逐项核对:
1. 首屏: chips、预览图、stem 卡(推理完成后出 grid 图)、呼吸箭头;
2. 向下滚动: 每个 stage 屏进入视口后网格图淡入、cos 数字 count-up、直方图生长;
3. 右侧圆点随滚动高亮, 点击可跳转; 顶部进度条随滚动填充;
4. 末屏: 双路径预测、概率条、指标、takeaway;
5. 重新「抽图」: 各屏回骨架态并重新推理;
6. 推理中快速滚动到 stage 屏: 骨架脉冲等待, 数据到达后自动揭示。

- [ ] **Step 4: Commit**

```bash
git add demo/web/app.js
git commit -m "rewrite demo frontend flow: auto-infer + scroll-triggered stage reveal"
```

---

### Task 5: 收尾 — 旧设计文档标注 + 全量回归 + 真机/降级双模式自审

**Files:**
- Modify: `demo/docs/frontend-design.md:1-5` (头部加 superseded 标注)

**Interfaces:**
- Consumes: Task 1-4 全部产物。
- Produces: 无。

- [ ] **Step 1: 标注旧设计文档**

`demo/docs/frontend-design.md` 第 3-4 行之间(标题下引用块)插入一行:

```markdown
> ⚠ 2026-07-18: 交互形态(单屏三栏)已被 frontend-scroll-design.md 的单栏滚动叙事取代;
> 本文档仅视觉语言与 grid_b64 渲染规则部分仍然有效。
```

- [ ] **Step 2: 全量回归**

Run: `cd Ltsimulator-test && python3 -m pytest demo/tests -q`
Expected: 全绿 (42 passed); 工作区无遗留改动。

- [ ] **Step 3: 双模式自审**

降级模式: Task 4 Step 2 的服务仍跑(或重启) → 浏览器完整滚一遍 7 屏, 按 1080p 截图自审美学/字号/对比度。
真机模式(若现场链路可用): 按 `demo/deploy.sh` 链路起真机远程 → 核对 health 灯变青、无降级 banner、逐屏揭示正常。
发现问题 → 修复后回到 Step 2 复跑。

- [ ] **Step 4: Commit**

```bash
git add demo/docs/frontend-design.md demo/web
git commit -m "finalize scroll-narrative demo: deprecate 3-column layout docs"
```

---

## Self-Review 记录

- Spec 覆盖: 7 屏结构(屏1 stem 卡、屏2-6、屏7)→ Task 3; 圆点导航+进度条 → Task 3/4; 自动推理+IO 揭示+骨架等待+重置 → Task 4; 5 个契约字段+MOPs 对账 → Task 1/2; 旧文档标注 → Task 5。末屏未含「全程误差总览图」(用户未选, 符合 spec)。
- 类型一致: `inject_comparison/compare_acts/LAYER_MOPS/REL_EDGES/check_comparison_fields` 在 Task 1/2 间一致; Task 3 的 DOM id/class 与 Task 4 的查询选择器逐一核对一致; `rel_err_hist.counts` 长度 = `len(edges)+1 = 9` 三处一致。
- 已知取舍: stage1 MOPs 按官方口径(MaxPool 后分辨率)计, 表内注释说明; `?autoinfer` 参数废弃(自动推理已是默认)。
