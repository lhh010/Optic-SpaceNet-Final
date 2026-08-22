"use strict";

const $ = (id) => document.getElementById(id);

const MODELS = [
  { id: 1, color: "#f472b6", label: "Model 1", cssClass: "text-m1" },
  { id: 2, color: "#a78bfa", label: "Model 2", cssClass: "text-m2" },
  { id: 3, color: "#22d3ee", label: "Model 3", cssClass: "text-m3" },
];

let currentImageB64 = null;
let currentLabel = null;
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
    '<div class="text-xs text-yellow-500 mt-2">⚠ 降级: fake-optical</div>' : "";

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
  inferRunning = true;
  $("btn-infer").disabled = true;
  $("infer-status").textContent = "推理中… 三模型并行请求";
  $("sec-summary").style.display = "none";

  const results = {};

  const promises = MODELS.map(async (m) => {
    renderWaiting(m.id);
    try {
      const res = await fetch("/api/compare-infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_b64: currentImageB64, model_id: m.id }),
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
  $("infer-status").textContent = "推理完成";

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
  const rows = [
    { label: "预测结果", key: (mid) => results[mid]?.pred || "-" },
    { label: "推理耗时", key: (mid) => results[mid] ? `${results[mid].latency_total_s.toFixed(2)}s` : "-" },
    { label: "引擎", key: (mid) => results[mid]?.engine || "-" },
    { label: "参数量", key: (mid) => metrics[mid] ? formatNum(metrics[mid].params) : "-" },
    { label: "MACs/张", key: (mid) => metrics[mid] ? formatNum(metrics[mid].macs_per_image) : "-" },
    { label: "光计算占比", key: (mid) => metrics[mid] ? `${(metrics[mid].optic_ratio * 100).toFixed(1)}%` : "-" },
    { label: "osim 精度", key: (mid) => metrics[mid] ? `${(metrics[mid].osim_acc * 100).toFixed(2)}%` : "-" },
    { label: "QAT 精度", key: (mid) => metrics[mid] ? `${(metrics[mid].qat_acc * 100).toFixed(2)}%` : "-" },
    { label: "真机单张耗时", key: (mid) => metrics[mid] ? `${metrics[mid].per_image_s}s` : "-" },
  ];

  tbody.innerHTML = rows.map((r) => `
    <tr class="border-b border-edge/50">
      <td class="py-2 px-2 text-slate-400">${r.label}</td>
      <td class="py-2 px-2 text-center text-m1">${r.key("1")}</td>
      <td class="py-2 px-2 text-center text-m2">${r.key("2")}</td>
      <td class="py-2 px-2 text-center text-m3">${r.key("3")}</td>
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
          <div class="flex items-center gap-2">
            <div class="text-[10px] text-slate-500 w-16 truncate">${l.name}</div>
            <div class="flex-1 h-4 bg-ink rounded overflow-hidden">
              <div class="h-full bar-grow rounded" style="width:${(l.latency_s / maxLatency * 100).toFixed(1)}%;background:${m.color};opacity:${l.where === 'optical' ? '0.8' : '0.4'};"></div>
            </div>
            <div class="text-[10px] text-slate-400 w-14 text-right">${l.latency_s.toFixed(3)}s</div>
            <div class="text-[9px] w-6 ${l.where === 'optical' ? 'text-photon' : 'text-elec'}">${l.where === 'optical' ? '光' : '电'}</div>
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
loadClasses();
