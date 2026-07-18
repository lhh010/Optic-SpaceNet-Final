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
