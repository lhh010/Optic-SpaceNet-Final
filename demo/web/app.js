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
  const seq = ++inferSeq;
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
    if (seq !== inferSeq) return;   // 不动新请求的锁 / 定时器
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
