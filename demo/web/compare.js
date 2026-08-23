"use strict";

const $ = (id) => document.getElementById(id);

const MODELS = [
  { id: 3, color: "#22d3ee", label: "Model 3", cssClass: "text-m3" },
  { id: 9, color: "#a78bfa", label: "M9", cssClass: "text-m9" },
  { id: 10, color: "#fbbf24", label: "M10", cssClass: "text-m10" },
];

const BACKENDS = {
  osimulator: {
    label: "本地 osimulator",
    help: "三种模型均通过本地 LT-Simulator 执行逐层光计算。",
  },
  gazelle_ssh: {
    label: "Gazelle SSH · uisrc@192.168.31.158",
    help: "三种模型均经 SSH 隧道访问板端 /matmul；板卡不可达时逐卡明确降级。",
  },
  gazelle_serial: {
    label: "Gazelle 串口",
    help: "串口 console 发现板卡 IP 后，三种模型访问板端 /matmul。",
  },
};

let currentImageB64 = null;
let currentLabel = null;
let currentBackend = "osimulator";
let inferRunning = false;
let elapsedTimers = {};

// ============================================================
//  Image selection (reuses /api/sample)
// ============================================================

async function loadClasses() {
  try {
    const res = await fetch("/api/sample?class=random");
    const data = await res.json();
    const select = $("class-select");
    select.innerHTML = '<option value="random">随机类别</option>';
    for (const cls of data.classes) {
      select.innerHTML += `<option value="${cls}">${cls}</option>`;
    }
    showImage(data.image_b64, data.label);
  } catch (e) {
    $("infer-status").textContent = "⚠ 后端连接失败";
  }
}

function showImage(b64, label) {
  currentImageB64 = b64;
  currentLabel = label;
  $("img-preview").innerHTML =
    `<img src="data:image/jpeg;base64,${b64}" class="w-full h-full object-cover">`;
  $("btn-infer").disabled = false;
  $("infer-status").textContent = label ? `类别: ${label} — 就绪` : "就绪";
}

$("btn-sample").addEventListener("click", async () => {
  const cls = $("class-select").value;
  const url = cls === "random" ? "/api/sample?random=true" : `/api/sample?class=${cls}`;
  try {
    const res = await fetch(url);
    const data = await res.json();
    showImage(data.image_b64, data.label);
    resetCards();
  } catch (e) {
    $("infer-status").textContent = "⚠ 抽图失败";
  }
});

$("file").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    const b64 = reader.result.split(",")[1];
    showImage(b64, null);
    resetCards();
  };
  reader.readAsDataURL(file);
});

// ============================================================
//  Inference
// ============================================================

function resetCards() {
  for (const m of MODELS) {
    $(`result-m${m.id}`).innerHTML =
      '<span class="text-slate-600 text-xs">等待推理…</span>';
    $(`card-m${m.id}`).classList.remove("skeleton");
  }
  $("sec-summary").style.display = "none";
}

function startElapsedTimer(modelId) {
  const startTime = performance.now();
  const el = document.createElement("span");
  el.className = "elapsed-timer text-slate-500 text-xs";
  el.textContent = "0.0s";
  elapsedTimers[modelId] = setInterval(() => {
    const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
    el.textContent = `${elapsed}s`;
  }, 100);
  return el;
}

function stopElapsedTimer(modelId) {
  if (elapsedTimers[modelId]) {
    clearInterval(elapsedTimers[modelId]);
    delete elapsedTimers[modelId];
  }
}

function renderWaiting(modelId) {
  const card = $(`card-m${modelId}`);
  card.classList.add("skeleton");
  const container = $(`result-m${modelId}`);
  container.innerHTML = `
    <div class="flex flex-col items-center gap-3">
      <div class="w-6 h-6 border-2 border-photon border-t-transparent rounded-full spinner"></div>
      <div class="text-xs text-slate-400">推理中…</div>
      <div id="timer-m${modelId}"></div>
    </div>`;
  const timerEl = startElapsedTimer(modelId);
  $(`timer-m${modelId}`).appendChild(timerEl);
}

function renderResult(modelId, result) {
  stopElapsedTimer(modelId);
  const card = $(`card-m${modelId}`);
  card.classList.remove("skeleton");
  const container = $(`result-m${modelId}`);
  const m = MODELS.find((x) => x.id === modelId);

  const top3 = Object.entries(result.probs).slice(0, 3);
  const maxProb = top3[0] ? top3[0][1] : 0;

  const correct = currentLabel ? result.pred === currentLabel : null;
  const verdictHtml = correct === null ? "" :
    correct ? '<span class="text-green-400 text-xs">✓ 正确</span>' :
              '<span class="text-red-400 text-xs">✗ 错误</span>';

  const degradedHtml = result.degraded ?
    `<div class="text-xs text-yellow-500 mt-2">⚠ ${result.target || BACKENDS[currentBackend].label} 不可用 → ${result.engine}</div>` : "";

  container.innerHTML = `
    <div class="w-full fade-in">
      <div class="flex items-center justify-between mb-3">
        <div class="text-lg font-bold" style="color:${m.color}">${result.pred}</div>
        ${verdictHtml}
      </div>
      <div class="flex flex-col gap-1.5 mb-4">
        ${top3.map(([cls, prob]) => `
          <div class="flex items-center gap-2">
            <div class="text-[10px] text-slate-400 w-28 truncate">${cls}</div>
            <div class="flex-1 h-3 bg-ink rounded overflow-hidden">
              <div class="h-full bar-grow rounded" style="width:${(prob / maxProb * 100).toFixed(1)}%;background:${m.color};opacity:0.7;"></div>
            </div>
            <div class="text-[10px] text-slate-300 w-10 text-right">${(prob * 100).toFixed(1)}%</div>
          </div>`).join("")}
      </div>
      <div class="border-t border-edge pt-3 grid grid-cols-2 gap-2 text-xs">
        <div>
          <div class="text-slate-500">总耗时</div>
          <div class="font-bold" style="color:${m.color}">${result.latency_total_s.toFixed(2)}s</div>
        </div>
        <div>
          <div class="text-slate-500">引擎</div>
          <div class="text-slate-300">${result.engine}</div>
        </div>
      </div>
      ${degradedHtml}
    </div>`;
}

function renderError(modelId, err) {
  stopElapsedTimer(modelId);
  const card = $(`card-m${modelId}`);
  card.classList.remove("skeleton");
  $(`result-m${modelId}`).innerHTML =
    `<div class="text-red-400 text-xs text-center">推理失败: ${err}</div>`;
}

async function runInference() {
  if (!currentImageB64 || inferRunning) return;
  const backend = currentBackend;
  inferRunning = true;
  $("btn-infer").disabled = true;
  document.querySelectorAll('input[name="compare-backend"]').forEach((el) => { el.disabled = true; });
  $("compare-backend-switch").classList.add("opacity-50", "pointer-events-none");
  $("infer-status").textContent = `三模型并行请求中：${BACKENDS[backend].label}`;
  $("sec-summary").style.display = "none";

  const results = {};

  const promises = MODELS.map(async (m) => {
    renderWaiting(m.id);
    try {
      const res = await fetch("/api/compare-infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_b64: currentImageB64, model_id: m.id, backend }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || res.statusText);
      }
      const data = await res.json();
      results[m.id] = data;
      renderResult(m.id, data);
    } catch (e) {
      renderError(m.id, e.message);
    }
  });

  await Promise.all(promises);
  inferRunning = false;
  $("btn-infer").disabled = false;
  document.querySelectorAll('input[name="compare-backend"]').forEach((el) => { el.disabled = false; });
  $("compare-backend-switch").classList.remove("opacity-50", "pointer-events-none");
  $("infer-status").textContent = `推理完成 · ${BACKENDS[backend].label}`;

  if (Object.keys(results).length > 0) {
    renderSummary(results);
  }
}

$("btn-infer").addEventListener("click", runInference);

// ============================================================
//  Summary Panel
// ============================================================

async function loadMetrics() {
  try {
    const res = await fetch("/api/compare-metrics");
    return await res.json();
  } catch {
    return null;
  }
}

async function renderSummary(results) {
  const metrics = await loadMetrics();
  if (!metrics) return;

  $("sec-summary").style.display = "block";

  const tbody = $("summary-tbody");
  const pctOrDash = (v) => v == null ? "-" : `${(v * 100).toFixed(2)}%`;
  const rows = [
    { label: "预测结果", key: (mid) => results[mid]?.pred || "-" },
    { label: "本次耗时", key: (mid) => results[mid] ? `${results[mid].latency_total_s.toFixed(2)}s` : "-" },
    { label: "本次引擎", key: (mid) => results[mid]?.engine || "-" },
    { label: "执行目标", key: (mid) => results[mid]?.target || "-" },
    { label: "参数量", key: (mid) => metrics[mid] ? formatNum(metrics[mid].params) : "-" },
    { label: "MACs/张", key: (mid) => metrics[mid] ? formatNum(metrics[mid].macs_per_image) : "-" },
    { label: "干净/QAT 参考", key: (mid) => pctOrDash(metrics[mid]?.reference_acc) },
    { label: "osimulator 全量", key: (mid) => pctOrDash(metrics[mid]?.osim_acc) },
    { label: "Gazelle 真机全量", key: (mid) => {
        const m = metrics[mid];
        return m?.hardware_acc == null ? "-" :
          `${pctOrDash(m.hardware_acc)} (n=${m.hardware_n})`;
      } },
    { label: "真机−干净 gap", key: (mid) =>
        metrics[mid]?.hardware_gap_pt == null ? "-" :
          `${metrics[mid].hardware_gap_pt.toFixed(2)} pt` },
  ];

  tbody.innerHTML = rows.map((r) => `
    <tr class="border-b border-edge/50">
      <td class="py-2 px-2 text-slate-400">${r.label}</td>
      <td class="py-2 px-2 text-center text-m3">${r.key("3")}</td>
      <td class="py-2 px-2 text-center text-m9">${r.key("9")}</td>
      <td class="py-2 px-2 text-center text-m10">${r.key("10")}</td>
    </tr>`).join("");

  renderLatencyChart(results);
}

function renderLatencyChart(results) {
  const chart = $("latency-chart");
  chart.innerHTML = "";

  for (const m of MODELS) {
    const r = results[m.id];
    if (!r || !r.layers) continue;

    const maxLatency = Math.max(...r.layers.map((l) => l.latency_s), 0.001);
    const section = document.createElement("div");
    section.innerHTML = `
      <div class="text-xs font-bold mb-2" style="color:${m.color}">${m.label} (总: ${r.latency_total_s.toFixed(2)}s)</div>
      <div class="flex flex-col gap-1">
        ${r.layers.map((l) => `
          <div class="border-b border-edge/40 pb-2 last:border-0">
            <div class="flex items-center gap-2">
              <div class="text-[10px] text-slate-300 w-14 truncate">${l.name}</div>
              <div class="flex-1 h-3 bg-ink rounded overflow-hidden">
                <div class="h-full bar-grow rounded" style="width:${(l.latency_s / maxLatency * 100).toFixed(1)}%;background:${m.color};opacity:${l.where === 'optical' ? '0.8' : '0.4'};"></div>
              </div>
              <div class="text-[10px] text-slate-400 w-14 text-right">${l.latency_s.toFixed(3)}s</div>
              <div class="text-[9px] w-6 ${l.where === 'optical' ? 'text-photon' : 'text-elec'}">${l.where === 'optical' ? '光' : '电'}</div>
            </div>
            <div class="text-[10px] text-slate-500 mt-1 ml-16">${l.spec || ""}</div>
            ${l.analysis ? `<div class="text-[10px] text-slate-600 mt-1 ml-16 leading-4">${l.analysis}</div>` : ""}
          </div>`).join("")}
      </div>`;
    chart.appendChild(section);
  }
}

function formatNum(n) {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`;
  return n.toString();
}

// ============================================================
//  Init
// ============================================================
function applyBackendSelection() {
  const checked = document.querySelector('input[name="compare-backend"]:checked');
  currentBackend = checked ? checked.value : "osimulator";
  const cfg = BACKENDS[currentBackend];
  $("compare-backend-help").textContent = cfg.help;
  for (const m of MODELS) {
    $(`target-m${m.id}`).textContent = `执行目标：${cfg.label}`;
  }
  resetCards();
  if (currentImageB64) {
    $("infer-status").textContent = `类别: ${currentLabel || "上传图"} — 已选择 ${cfg.label}`;
  }
}

document.querySelectorAll('input[name="compare-backend"]').forEach((el) => {
  el.addEventListener("change", applyBackendSelection);
});

const requestedBackend = new URLSearchParams(location.search).get("backend");
if (requestedBackend && BACKENDS[requestedBackend]) {
  const requestedRadio = document.querySelector(
    `input[name="compare-backend"][value="${requestedBackend}"]`);
  if (requestedRadio) requestedRadio.checked = true;
}
applyBackendSelection();
loadClasses();
