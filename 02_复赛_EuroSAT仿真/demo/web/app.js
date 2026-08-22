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

/* 逐层结构静态表 (镜像后端 demo/server/model_trace.py 的 EXPECTED_SHAPES /
   LAYER_SPECS / LAYER_WHERE)。纯前端可视化, 不走 API, 运行时恒定。
   shape: conv 层 [C,H,W]; fc 层 [N] (1 维体)。ops: 该层算子链, note 为核/步长标注。 */
const ARCH = {
  stem:   { where: "electronic", in: [3, 64, 64],  out: [8, 64, 64],
            ops: [{ k: "Conv", note: "1×1" }, { k: "BN" }, { k: "ReLU" }] },
  stage1: { where: "optical", in: [8, 64, 64], out: [16, 16, 16],
            ops: [{ k: "Conv", note: "2×2 /s2" }, { k: "BN" }, { k: "ReLU" }, { k: "MaxPool", note: "2×2" }] },
  stage2: { where: "optical", in: [16, 16, 16], out: [32, 8, 8],
            ops: [{ k: "Conv", note: "2×2 /s2" }, { k: "BN" }, { k: "ReLU" }] },
  stage3: { where: "optical", in: [32, 8, 8], out: [16, 8, 8],
            ops: [{ k: "Conv", note: "1×1" }, { k: "BN" }, { k: "ReLU" }] },
  fc1:    { where: "optical", in: [1024], out: [256],
            ops: [{ k: "Linear" }, { k: "ReLU" }] },
  fc2:    { where: "optical", in: [256], out: [10],
            ops: [{ k: "Linear" }] },
};

/* 跨 6 层归一化的尺寸基准: 让体块大小在屏间可比 (8×64×64 大 → 10 小)。 */
const ARCH_MAX_SPATIAL = 64;    // 最大 H/W (stem/stage1 输入)
const ARCH_MAX_CHAN = 32;       // 最大通道 (stage2 输出)
const ARCH_MAX_FC = 1024;       // 最大 fc 宽度 (fc1 输入)

/* 一个 tensor 体块 (isometric 堆叠板): 正面 = 空间 H×W, 偏移深度面 = 通道堆叠感。
   conv 体 [C,H,W]: 正面边长 ∝ H/W, 深度 ∝ C; fc 体 [N]: 退化成 1 维竖条, 高 ∝ N。 */
function archVolume(shape, x, cy, color) {
  const stroke = color, fill = "#0b1120";
  if (shape.length === 1) {                       // fc: 1 维竖条
    const n = shape[0];
    const h = 24 + (n / ARCH_MAX_FC) * 66;         // 24..90
    const w = 26;
    const y = cy - h / 2;
    return `<g>
      <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="3"
            fill="${fill}" stroke="${stroke}" stroke-width="1.5"/>
      <text x="${x + w / 2}" y="${cy}" text-anchor="middle" dominant-baseline="central"
            font-size="12" fill="${stroke}" font-weight="700">${n}</text>
    </g>`;
  }
  const [c, hh] = shape;                            // conv: [C,H,W] (H==W)
  const side = 34 + (hh / ARCH_MAX_SPATIAL) * 62;   // 34..96 正面边长
  const depth = 6 + (c / ARCH_MAX_CHAN) * 20;       // 6..26 深度偏移(通道感)
  const x0 = x, y0 = cy - side / 2;
  const dx = depth, dy = -depth;
  // 顶面 + 侧面(深度) + 正面, 组成 isometric 板
  return `<g>
    <polygon points="${x0},${y0} ${x0 + side},${y0} ${x0 + side + dx},${y0 + dy} ${x0 + dx},${y0 + dy}"
             fill="#0e1a30" stroke="${stroke}" stroke-width="1.2" opacity="0.9"/>
    <polygon points="${x0 + side},${y0} ${x0 + side},${y0 + side} ${x0 + side + dx},${y0 + side + dy} ${x0 + side + dx},${y0 + dy}"
             fill="#0a1526" stroke="${stroke}" stroke-width="1.2" opacity="0.9"/>
    <rect x="${x0}" y="${y0}" width="${side}" height="${side}" rx="2"
          fill="${fill}" stroke="${stroke}" stroke-width="1.5"/>
    <text x="${x0 + side / 2}" y="${y0 + side + 15}" text-anchor="middle"
          font-size="11" fill="#94a3b8">${c}×${hh}×${hh}</text>
    <text x="${x0 + side / 2}" y="${y0 + side / 2}" text-anchor="middle" dominant-baseline="central"
          font-size="12" fill="${stroke}" font-weight="700">${c}ch</text>
  </g>`;
}

/* 算子药丸链: 每个 op 一个 pill, 竖直堆叠居中; 可选 note (核/步长)。 */
function archOps(ops, cx, cy, color) {
  const pw = 116, ph = 22, gap = 8;
  const totalH = ops.length * ph + (ops.length - 1) * gap;
  let y = cy - totalH / 2;
  const pills = ops.map((op) => {
    const label = op.note ? `${op.k} ${op.note}` : op.k;
    const g = `<g>
      <rect x="${cx - pw / 2}" y="${y}" width="${pw}" height="${ph}" rx="11"
            fill="#0b1120" stroke="${color}" stroke-width="1.2" opacity="0.95"/>
      <text x="${cx}" y="${y + ph / 2}" text-anchor="middle" dominant-baseline="central"
            font-size="11" fill="${color}">${label}</text>
    </g>`;
    y += ph + gap;
    return g;
  }).join("");
  return pills;
}

/* 箭头连接线 (输入体 → 算子 → 输出体)。 */
function archArrow(x1, x2, cy, color) {
  return `<line x1="${x1}" y1="${cy}" x2="${x2 - 7}" y2="${cy}"
            stroke="${color}" stroke-width="1.4" opacity="0.6"/>
    <polygon points="${x2 - 7},${cy - 4} ${x2},${cy} ${x2 - 7},${cy + 4}"
             fill="${color}" opacity="0.6"/>`;
}

/* 整宽 tensor-flow band: [输入体] → [算子链] → [输出体]。
   optical 层用 photon 青, stem(电层)用 elec 灰。三组 .arch-part 错峰揭示。 */
function archSVG(name) {
  const a = ARCH[name];
  if (!a) return "";
  const color = a.where === "optical" ? "#22d3ee" : "#94a3b8";
  const W = 760, H = 150, cy = 74;
  const inX = 40, opsCx = W / 2, outX = W - 150;
  return `<svg class="arch-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet"
               role="img" aria-label="${name} 结构图">
    <g class="arch-part p1">${archVolume(a.in, inX, cy, color)}</g>
    <g class="arch-part p1">${archArrow(inX + 130, opsCx - 70, cy, color)}</g>
    <g class="arch-part p2">${archOps(a.ops, opsCx, cy, color)}</g>
    <g class="arch-part p3">${archArrow(opsCx + 70, outX, cy, color)}</g>
    <g class="arch-part p3">${archVolume(a.out, outX, cy, color)}</g>
  </svg>`;
}

let current = null;        // {image_b64, label|null}
let inferData = null;      // /api/infer 响应体
let inferState = "idle";   // idle | running | done | error
let inferSeq = 0;          // 单调递增请求序号: 旧请求在途时新请求生效, 旧响应一律丢弃
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
      <div class="card reveal-item mb-5 stage-arch"></div>
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
  sec.querySelector(".stage-arch").innerHTML = archSVG(name);
  sec.querySelector(".stage-fig").innerHTML = ol.grid_b64
    ? `<img class="grid-img rounded border border-edge w-full" alt="${name}"
         src="data:image/png;base64,${ol.grid_b64}">`
    : `<div class="text-xs text-slate-500 py-8 text-center">该层 feature map 不可用</div>`;
  if (ol.cos_sim == null || ol.rel_err_hist == null) {
    sec.querySelector(".stage-cos").textContent = "-";
    sec.querySelector(".stage-maxabs b").textContent = "-";
    sec.querySelector(".stage-hist").innerHTML =
      `<div class="text-xs text-slate-500 py-8 text-center">该层对比数据不可用</div>`;
    sec.querySelector(".stage-theo").textContent =
      ol.theoretical_s != null ? fmtLat(ol.theoretical_s) : "-";
    sec.querySelector(".stage-mops").textContent =
      ol.mops != null ? `${ol.mops} MOPs` : "-";
    sec.querySelector(".stage-elec").textContent = fmtLat(fl.latency_s);
    return;
  }
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
  const seq = ++inferSeq;
  resetReveal();
  lockInputs(true);
  inferState = "running";
  const t0 = performance.now();
  const myTimer = setInterval(() => {
    $("infer-status").textContent =
      `光计算推理中… ${((performance.now() - t0) / 1000).toFixed(1)}s`;
  }, 100);
  try {
    const body = await j(await fetch("/api/infer", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(current),
    }));
    if (seq !== inferSeq) return;   // 已被更新的请求取代, 丢弃旧响应
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
    if (seq !== inferSeq) return;
    inferState = "error";
    showBanner("error", `推理失败: ${e.message}`);
    $("infer-status").textContent = "";
  } finally {
    clearInterval(myTimer);         // 自己的定时器总能安全清除, 即使已过期
    if (seq !== inferSeq) return;   // 不动新请求的锁
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
$("stem-arch").innerHTML = archSVG("stem");
observeScreens();
refreshHealth();
loadMetrics().catch(() => {});
loadSample().catch((e) => showBanner("error", `抽图失败: ${e.message}`));
