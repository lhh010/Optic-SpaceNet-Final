# 前端美学版实现计划 — Model 3 光计算演示

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `demo/web/` 占位页替换为深色光子科技感的单屏三栏 Dashboard, 后端新增 `render.py` 按契约渲染逐层 feature map PNG (`grid_b64`)。

**Architecture:** 本地 FastAPI 在 `/api/infer` 聚合阶段把 fp32/optical 两条 PathResult 的同层激活做共享 min/max 归一化并渲成一张 PNG 注入每层 `grid_b64`; 前端零构建 (本地化 Tailwind Play CDN + vanilla JS) 只负责展示。契约见 `demo/docs/api.md`, 设计见 `demo/docs/frontend-design.md`。

**Tech Stack:** Python (numpy/PIL) + FastAPI; HTML + Tailwind Play CDN (本地化) + vanilla JS; pytest。

## Global Constraints

- 契约字段名以 `demo/docs/api.md` 为准 (`grid_b64`, 每条 PathResult 的 layers 都要注入), 不得改名。
- 视觉/布局以 `demo/docs/frontend-design.md` 为准: 深空蓝黑底 (#05070f), 光=青 #22d3ee/金 #fbbf24, 电=灰蓝 #64748b, 中文 UI, 单屏三栏, min-width 1280px。
- 运行时**零外网请求**: 所有前端资产必须在 `demo/web/vendor/` 本地化。
- 不改 `src/` 实验代码, 不改 `demo/remote/optic_server.py`; 仅新增 `demo/server/render.py`、改 `demo/server/app.py`、重写 `demo/web/`、新增测试。
- 远程 optic_server 不返回 `grid_b64` — 渲染只在本地后端发生, 因此 `demo/tests/contract.py` 保持不变 (grid 检查只写在 test_app.py)。
- 测试一律从 repo root 运行: `cd Ltsimulator-test && python3 -m pytest demo/tests -q`。
- Python 代码风格随 `demo/server/` 现有文件 (英文 docstring, 双引号)。
- 每 Task 结束独立 commit (git 提交信息小写英文, 随仓库习惯如 "add model3 optical inference demo backend")。

---

### Task 1: demo/server/render.py — 服务端 feature map 渲染

**Files:**
- Create: `demo/server/render.py`
- Test: `demo/tests/test_render.py`

**Interfaces:**
- Consumes: `demo.server.inference_local.encode_act_b64` (仅测试用), api.md 的 `act_b64` 格式 (np.savez float16, 键 "act")。
- Produces:
  - `render.decode_act(act_b64: str) -> np.ndarray` (float16, 形状同 Layer.shape)
  - `render.render_layer_png(name: str, opt_act: np.ndarray, el_act: np.ndarray) -> bytes` (PNG)
  - `render.inject_grids(fp32: dict, optical: dict) -> None` (就地给两条 PathResult 的每层设 `grid_b64`, 同层两路径值相同)
  - 常量 `GRID=122`, `SEP=2`, `BAR_W=256`, `BAR_H=10` (测试断言尺寸用)

- [ ] **Step 1: 写失败测试** `demo/tests/test_render.py`

```python
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd Ltsimulator-test && python3 -m pytest demo/tests/test_render.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'demo.server.render'`

- [ ] **Step 3: 实现** `demo/server/render.py`

```python
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd Ltsimulator-test && python3 -m pytest demo/tests/test_render.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
cd Ltsimulator-test
git add demo/server/render.py demo/tests/test_render.py
git commit -m "add server-side feature-map renderer (layer grid_b64)"
```

---

### Task 2: app.py 接入 inject_grids + 契约测试

**Files:**
- Modify: `demo/server/app.py` (import 行与 infer())
- Test: `demo/tests/test_app.py` (追加一个测试)

**Interfaces:**
- Consumes: `render.inject_grids(fp32, optical)` (Task 1)。
- Produces: `/api/infer` 响应中 fp32/optical 的每层都含 `grid_b64` (PNG b64); conv 层 PNG 尺寸 (246,122), fc 层 (256,22)。

- [ ] **Step 1: 追加失败测试** — 在 `demo/tests/test_app.py` 末尾加:

```python
def test_infer_layers_carry_decodable_grids(client, monkeypatch, sample_image):
    monkeypatch.setenv("OPTIC_REMOTE_URL", _dead_url())
    r = client.post("/api/infer", json={
        "image_b64": sample_image["image_b64"], "label": sample_image["label"]})
    assert r.status_code == 200
    body = r.json()
    for path in (body["fp32"], body["optical"]):
        assert [l["name"] for l in path["layers"]] == contract.LAYER_NAMES
        for layer, shape in zip(path["layers"], contract.LAYER_SHAPES):
            img = Image.open(io.BytesIO(base64.b64decode(layer["grid_b64"])))
            assert img.format == "PNG"
            expected = (246, 122) if len(shape) == 3 else (256, 22)
            assert img.size == expected, layer["name"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd Ltsimulator-test && python3 -m pytest demo/tests/test_app.py::test_infer_layers_carry_decodable_grids -q`
Expected: FAIL, `KeyError: 'grid_b64'`

- [ ] **Step 3: 修改 app.py**

import 行 (第 36 行附近) 改为:

```python
from demo.server import inference_local, remote_client, render  # noqa: E402
```

`infer()` 中 `correct = ...` 块之后、`return` 之前插入:

```python
    render.inject_grids(fp32, optical)
```

- [ ] **Step 4: 跑全量测试确认通过**

Run: `cd Ltsimulator-test && python3 -m pytest demo/tests -q`
Expected: 35 passed (34 旧 + 1 新)

- [ ] **Step 5: Commit**

```bash
cd Ltsimulator-test
git add demo/server/app.py demo/tests/test_app.py
git commit -m "inject grid_b64 into /api/infer layer results"
```

---

### Task 3: 前端 vendor 资产本地化

**Files:**
- Create: `demo/web/vendor/tailwind.play.js`
- Create: `demo/web/vendor/jetbrains-mono-400.woff2`
- Create: `demo/web/vendor/jetbrains-mono-700.woff2`
- Create: `demo/web/vendor/fonts.css`

**Interfaces:**
- Consumes: 无 (公网下载, 一次性 vendoring)。
- Produces: `vendor/tailwind.play.js` (Task 4 index.html `<script src>`), `vendor/fonts.css` 暴露字体族 `"JetBrains Mono"` (Task 4 tailwind.config fontFamily 引用)。

- [ ] **Step 1: 下载资产**

```bash
cd Ltsimulator-test/demo/web
mkdir -p vendor
curl -fsSL -o vendor/tailwind.play.js "https://cdn.tailwindcss.com/3.4.16"
curl -fsSL -o vendor/jetbrains-mono-400.woff2 "https://cdn.jsdelivr.net/npm/@fontsource/jetbrains-mono@5/files/jetbrains-mono-latin-400-normal.woff2"
curl -fsSL -o vendor/jetbrains-mono-700.woff2 "https://cdn.jsdelivr.net/npm/@fontsource/jetbrains-mono@5/files/jetbrains-mono-latin-700-normal.woff2"
ls -la vendor/
```

Expected: tailwind.play.js > 100KB; 两个 woff2 各 > 10KB; curl 无错误退出码。
若 jsdelivr 不可达, 备用: `https://fastly.jsdelivr.net/npm/@fontsource/jetbrains-mono@5/files/...`。

- [ ] **Step 2: 写字体声明** `demo/web/vendor/fonts.css`

```css
/* Vendored JetBrains Mono (fontsource, OFL) — zero external requests. */
@font-face {
  font-family: "JetBrains Mono";
  src: url("jetbrains-mono-400.woff2") format("woff2");
  font-weight: 400;
  font-style: normal;
  font-display: swap;
}
@font-face {
  font-family: "JetBrains Mono";
  src: url("jetbrains-mono-700.woff2") format("woff2");
  font-weight: 700;
  font-style: normal;
  font-display: swap;
}
```

- [ ] **Step 3: 验证能被 StaticFiles 伺服**

```bash
cd Ltsimulator-test && python3 -m uvicorn demo.server.app:app --port 8000 &
sleep 3
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/vendor/tailwind.play.js
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/vendor/fonts.css
kill %1
```

Expected: 两个 200。

- [ ] **Step 4: Commit**

```bash
cd Ltsimulator-test
git add demo/web/vendor/
git commit -m "vendor tailwind play cdn and jetbrains mono locally"
```

---

### Task 4: demo/web/index.html — 单屏三栏骨架与主题

**Files:**
- Create (覆盖占位页): `demo/web/index.html`

**Interfaces:**
- Consumes: `vendor/tailwind.play.js`, `vendor/fonts.css` (Task 3)。
- Produces: DOM id 契约供 Task 5 的 app.js 使用: `health-dot, health-text, banner, img, label, index, class-select, btn-sample, file, btn-infer, stopwatch, layers, pred-fp32, pred-opt, lat-fp32, lat-opt, verdict, probs, metrics`。
- 页面文案中文; 骨架含静态占位, 无 JS 也能看清布局。

- [ ] **Step 1: 写 index.html (完整替换占位页)**

```html
<!DOCTYPE html>
<!-- Model 3 光计算演示 — 美学版 (demo/docs/frontend-design.md)。
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
  /* 辉光与进度脉冲 — Tailwind 之外的少量自定义 CSS */
  .glow-photon { text-shadow: 0 0 12px rgba(34, 211, 238, .55); }
  @keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(34, 211, 238, .5); }
    50%      { box-shadow: 0 0 18px 4px rgba(34, 211, 238, .35); }
  }
  .inferring { animation: pulse-glow 1s ease-in-out infinite; }
  .bar-photon { background: linear-gradient(90deg, #0891b2, #22d3ee);
                box-shadow: 0 0 8px rgba(34, 211, 238, .45); }
  .bar-elec { background: #475569; }
  .grid-img { image-rendering: pixelated; }
</style>
</head>
<body class="bg-ink text-slate-200 font-mono min-w-[1280px] min-h-screen">
<div class="max-w-[1600px] mx-auto px-5 py-4">

  <header>
    <div class="flex items-center gap-4">
      <h1 class="text-xl font-bold tracking-wide">
        <span class="text-photon glow-photon">⚡ 光计算推理演示</span>
        <span class="text-slate-400 text-sm ml-2 font-normal">Model 3 · SpaceNet V2 + KD · int8</span>
      </h1>
      <div class="flex items-center gap-2 ml-auto">
        <span id="health-dot" class="w-2.5 h-2.5 rounded-full bg-slate-600"></span>
        <span id="health-text" class="text-xs text-slate-400">连接中…</span>
      </div>
    </div>
    <div class="flex gap-2 mt-3">
      <span class="chip">光算占比 <b class="text-photon">90.65%</b></span>
      <span class="chip">算力削减 <b class="text-photon">150×</b></span>
      <span class="chip">osim 全量 <b class="text-photon">90.28%</b> (n=5400)</span>
      <span class="chip">硬件对齐 <b class="text-photon">99.6%</b></span>
      <span class="chip">int8 val <b class="text-photon">91.83%</b></span>
    </div>
  </header>

  <div id="banner" class="hidden"></div>

  <main class="grid grid-cols-[280px_minmax(0,1fr)_330px] gap-4 mt-4 items-start">

    <!-- 左栏: 输入 -->
    <section class="card flex flex-col gap-3">
      <h2 class="text-sm text-slate-400">输入</h2>
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
      <button id="btn-infer" disabled
              class="btn bg-photon text-ink hover:bg-cyan-300 disabled:opacity-40 text-base py-3 mt-1">
        ▶ 开始推理
      </button>
      <div id="stopwatch" class="hidden text-center text-photon text-sm">光计算推理中… 0.0s</div>
    </section>

    <!-- 中栏: 逐层对比 (主角) -->
    <section class="card">
      <h2 class="text-sm text-slate-400 mb-3">
        逐层对比 <span class="text-photon font-bold">光</span> |
        <span class="text-elec font-bold">电</span>
        <span class="text-xs text-slate-500">(同层共享归一化; conv 层 光左|电右, fc 层 光上|电下)</span>
      </h2>
      <div id="layers" class="flex flex-col gap-3">
        <div class="text-slate-500 text-sm">点击「开始推理」后展示 6 层 feature map</div>
      </div>
    </section>

    <!-- 右栏: 预测对比 + 指标面板 -->
    <section class="flex flex-col gap-4">
      <div class="card">
        <h2 class="text-sm text-slate-400 mb-3">预测对比</h2>
        <div class="grid grid-cols-2 gap-3 text-center">
          <div>
            <div class="text-xs text-elec mb-1">FP32 电计算</div>
            <div id="pred-fp32" class="text-lg font-bold">-</div>
            <div id="lat-fp32" class="text-[10px] text-slate-500"></div>
          </div>
          <div>
            <div class="text-xs text-photon mb-1">int8 光计算</div>
            <div id="pred-opt" class="text-lg font-bold text-photon glow-photon">-</div>
            <div id="lat-opt" class="text-[10px] text-slate-500"></div>
          </div>
        </div>
        <div id="verdict" class="text-center text-sm mt-2 min-h-[20px]"></div>
        <div id="probs" class="mt-3 flex flex-col gap-1.5"></div>
      </div>
      <div class="card">
        <h2 class="text-sm text-slate-400 mb-2">指标面板</h2>
        <div id="metrics" class="text-xs leading-6"></div>
      </div>
    </section>
  </main>
</div>
<script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: 起服务并验证骨架渲染**

```bash
cd Ltsimulator-test && python3 -m uvicorn demo.server.app:app --port 8000 &
sleep 3
curl -s http://127.0.0.1:8000/ | grep -o "光计算推理演示"
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu \
  --screenshot=/tmp/demo-skeleton.png --window-size=1600,1000 \
  --virtual-time-budget=8000 http://127.0.0.1:8000/
```

Expected: grep 命中; 截图生成。用 ReadMediaFile 查看 /tmp/demo-skeleton.png: 深色底、三栏、chips、左栏抽图已自动加载 (app.js 尚不存在, 暂为静态骨架 + 无数据属正常 — 本步只核布局与配色; 若 Chrome 不存在则改手动浏览器核对)。先不 kill 服务 (Task 5 继续用)。

- [ ] **Step 3: Commit**

```bash
cd Ltsimulator-test
git add demo/web/index.html
git commit -m "rewrite demo page skeleton with photon-dark three-column layout"
```

---

### Task 5: demo/web/app.js — 数据流与渲染

**Files:**
- Create: `demo/web/app.js`

**Interfaces:**
- Consumes: Task 4 的 DOM id 契约; `GET /api/health`, `GET /api/sample?class=`, `GET /api/metrics`, `POST /api/infer` (api.md; 每层含 `grid_b64`)。
- Produces: 页面全部交互; URL 参数 `?autoinfer=1` 表示抽图后自动推理 (无头截图自审/现场一键演示用)。

- [ ] **Step 1: 写 app.js**

```js
"use strict";
/* Model 3 光计算演示 — 页面数据流 (demo/docs/frontend-design.md)。
   数据契约: demo/docs/api.md (PathResult.layers[i].grid_b64 为光|电拼接 PNG)。 */

const $ = (id) => document.getElementById(id);
let current = null;   // {image_b64, label|null}
let timer = null;

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
  b.className = `mt-3 rounded-lg border px-4 py-2 text-sm ${styles[kind]}`;
  b.textContent = text;
}
function hideBanner() { $("banner").className = "hidden"; }

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
    ["真机单张耗时", `~${m.per_image_s}s`],
  ];
  $("metrics").innerHTML = rows.map(([k, v]) =>
    `<div class="flex justify-between border-b border-edge/60 py-0.5">
       <span class="text-slate-400">${k}</span><b class="text-slate-100">${v}</b>
     </div>`).join("");
}

function setImage(b64, label, index) {
  current = { image_b64: b64, label: label ?? null };
  $("img").src = "data:image/jpeg;base64," + b64;
  $("label").textContent = label ?? "(上传图, 无 ground truth)";
  $("index").textContent = index ?? "-";
  $("btn-infer").disabled = false;
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

function fmtLat(s) { return s >= 0.1 ? `${s.toFixed(2)}s` : `${(s * 1000).toFixed(1)}ms`; }

function renderPaths(body) {
  const { fp32, optical, meta } = body;
  $("pred-fp32").textContent = fp32.pred;
  $("pred-opt").textContent = optical.pred;
  $("lat-fp32").textContent = `total ${fmtLat(fp32.latency_total_s)}`;
  $("lat-opt").textContent = `total ${fmtLat(optical.latency_total_s)} · ${optical.engine}`;
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

  $("layers").innerHTML = optical.layers.map((ol, i) => {
    const fl = fp32.layers[i];
    const badge = ol.where === "optical"
      ? `<span class="text-[10px] px-1.5 py-0.5 rounded border border-photon/50 text-photon">光</span>`
      : `<span class="text-[10px] px-1.5 py-0.5 rounded border border-elec/60 text-elec">电</span>`;
    return `<div class="flex items-center gap-3 border-b border-edge/50 pb-2">
      <div class="w-[190px] shrink-0">
        <div class="text-sm font-bold">${ol.name} ${badge}</div>
        <div class="text-[10px] text-slate-500 leading-4 mt-0.5">${ol.spec}</div>
        <div class="text-[10px] text-slate-500">[${ol.shape.join(",")}]</div>
        <div class="text-xs mt-1">
          <span class="text-photon font-bold">⚡${fmtLat(ol.latency_s)}</span>
          <span class="text-slate-500"> / 电 ${fmtLat(fl.latency_s)}</span>
        </div>
      </div>
      <img class="grid-img rounded border border-edge" alt="${ol.name}"
           src="data:image/png;base64,${ol.grid_b64}">
    </div>`;
  }).join("");
}

$("btn-infer").onclick = async () => {
  if (!current) return;
  const btn = $("btn-infer");
  btn.disabled = true;
  btn.classList.add("inferring");
  btn.textContent = "◌ 光计算推理中…";
  $("stopwatch").classList.remove("hidden");
  const t0 = performance.now();
  timer = setInterval(() => {
    $("stopwatch").textContent =
      `光计算推理中… ${((performance.now() - t0) / 1000).toFixed(1)}s`;
  }, 100);
  try {
    const body = await j(await fetch("/api/infer", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(current),
    }));
    hideBanner();
    if (body.meta.degraded) {
      showBanner("warn", "⚠ 远程光计算不可用, 已降级为本地 fake 引擎 (meta.degraded=true)");
    }
    renderPaths(body);
  } catch (e) {
    showBanner("error", `推理失败: ${e.message}`);
  } finally {
    clearInterval(timer);
    $("stopwatch").classList.add("hidden");
    btn.classList.remove("inferring");
    btn.textContent = "▶ 开始推理";
    btn.disabled = false;
  }
};

$("btn-sample").onclick = () =>
  loadSample($("class-select").value).catch((e) => showBanner("error", e.message));

$("file").onchange = (ev) => {
  const f = ev.target.files[0];
  if (!f) return;
  const rd = new FileReader();
  rd.onload = () => { setImage(rd.result.split(",")[1], null, null); hideBanner(); };
  rd.readAsDataURL(f);
};

refreshHealth();
loadMetrics().catch(() => {});
const auto = new URLSearchParams(location.search).has("autoinfer");
loadSample()
  .then(() => { if (auto) $("btn-infer").click(); })
  .catch((e) => showBanner("error", `抽图失败: ${e.message}`));
```

- [ ] **Step 2: 无头截图自审 (降级模式全链路)**

Task 4 的 uvicorn 仍在跑 (远程隧道未建 → 自动走 fake 降级)。执行:

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu \
  --screenshot=/tmp/demo-fake.png --window-size=1600,1000 \
  --virtual-time-budget=30000 "http://127.0.0.1:8000/?autoinfer=1"
```

用 ReadMediaFile 查看 /tmp/demo-fake.png, 逐项核对 frontend-design.md:
- [ ] 黄色降级警示条可见 (meta.degraded=true)
- [ ] 中栏 6 行层卡片, conv 层为光|电拼接网格图, fc 层为条带, 有 ⚡耗时
- [ ] 右栏双 pred、概率双条形、verdict、指标面板
- [ ] 深色光子配色, 无未样式化的裸 HTML

如某要素缺失/错位: 修 app.js/index.html 后重截, 直到全部核对通过。

- [ ] **Step 3: curl 抽查 grid 通路**

```bash
curl -s http://127.0.0.1:8000/api/sample | python3 -c "
import json,sys,base64,io
from PIL import Image
s=json.load(sys.stdin())
print('sample:', s['label'], s['index'])"
```

Expected: 打印类别与 index (顺带确认服务正常)。页面侧 grid 已在 Step 2 截图中核对。

- [ ] **Step 4: Commit**

```bash
cd Ltsimulator-test
git add demo/web/app.js
git commit -m "add demo page data flow: health/sample/infer rendering"
```

---

### Task 6: 真机烟测 + 文档收尾

**Files:**
- Modify: `demo/docs/design.md:3` (状态行)
- Modify: `demo/docs/frontend-design.md:3` (状态行)

**Interfaces:**
- Consumes: 全部前序任务; `demo/deploy.sh` (远程服务重启, 幂等); SSH 隧道命令 (design.md 部署链路)。
- Produces: 真机模式下的页面验证记录; 文档状态更新。

- [ ] **Step 1: 重建 SSH 隧道并确认真机引擎**

```bash
curl -s --max-time 3 http://127.0.0.1:8765/health || true
# 若无响应: 先确认远程服务存活, 必要时重启
cd Ltsimulator-test && bash demo/deploy.sh
ssh -N -L 8765:172.17.0.2:8765 fdusc-cpu-135 &
sleep 3
curl -s http://127.0.0.1:8765/health
curl -s http://127.0.0.1:8000/api/health
```

Expected: 远程 `/health` 返回 `"engine":"gazelle-osim"`; 本地 `/api/health` 返回 `{"local":"ok","remote":"gazelle-osim"}`。
(若 deploy.sh 输出的容器 IP 不是 172.17.0.2, 以 `ssh fdusc-cpu-135 "docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' gazelle_sim"` 为准替换隧道目标。)

- [ ] **Step 2: 真机模式无头截图自审**

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu \
  --screenshot=/tmp/demo-real.png --window-size=1600,1000 \
  --virtual-time-budget=45000 "http://127.0.0.1:8000/?autoinfer=1"
```

用 ReadMediaFile 查看 /tmp/demo-real.png, 核对:
- [ ] 顶栏状态灯青色「真机光计算 gazelle-osim 已连接」, 无降级警示条
- [ ] 层卡片 ⚡耗时为秒级 (fc1 ~1.5s 量级), verdict 显示 ✓/✗
- [ ] 与 /tmp/demo-fake.png 对比, 布局一致

- [ ] **Step 3: 更新文档状态行**

`demo/docs/design.md` 第 3 行改为:

```
> 2026-07-17 · 状态: 后端与光计算 server 已完成并真机验证; 前端美学版已实现 (frontend-design.md)。
```

`demo/docs/frontend-design.md` 第 3 行改为:

```
> 2026-07-17 · 状态: 已实现并真机烟测通过 · 关联: design.md(总设计), api.md(契约)
```

- [ ] **Step 4: 全量测试回归 + Commit**

```bash
cd Ltsimulator-test && python3 -m pytest demo/tests -q
git add demo/docs/design.md demo/docs/frontend-design.md
git commit -m "mark demo frontend implemented after real-machine smoke test"
```

Expected: 35 passed; 工作区干净 (kill 掉 uvicorn 与隧道后台进程或留着供演示, 均可)。

---

## Self-Review 记录

- Spec 覆盖: 契约变更 grid_b64 → Task 1/2; 技术栈/零外网 → Task 3/4; 三栏布局/视觉 → Task 4; 数据流/进度指示/降级提示/上传 → Task 5; 测试 (render 单测 + app 契约 + 真机/降级截图自审) → Task 1/2/5/6。✔
- 类型一致性: `inject_grids(fp32, optical)` 在 Task 1 定义、Task 2 使用签名一致; 前端只读 `optical.layers[i].grid_b64` 与 `fp32.layers[i]` (同序 6 层, contract 保证)。✔
- 已知取舍: 前端无 JS 单元测试 (YAGNI, 演示单页), 以无头截图核对代替; `?autoinfer=1` 为此目的而设。
