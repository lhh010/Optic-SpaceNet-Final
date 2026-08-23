# -*- coding: utf-8 -*-
"""决赛真机演示后端 v3 — 路径B 板端 runner（浏览器 × 真机光计算）。

v3 变更 (2026-08-23): demo 从 pathA(HTTP matmul, 实测 ~10s/次, 单图90s+)
切到 pathB(板上 run_ds3_gazelle/run_mnist_gazelle/run_mnist_official 直调 compass,
~3.2s/张, M10 95.33% canonical 链)。由 board.py 经 SSH+sudo 触发板上 runner。

运行: uvicorn demo_hw.server:app --port 8100
环境: BOARD_HOST/PORT/USER/PASS (板上SSH), HW_CALIB(板上标量calib json名),
      HW_MODEL=model10|model9 决定 weights_m10_5400|weights_w075ds3
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
_OPTIC = os.path.join(_PKG, "03_决赛_EuroSAT真机", "opticspacenet")
_SRC = os.path.join(_PKG, "02_复赛_EuroSAT仿真", "src")
for _p in (_OPTIC, _SRC, os.path.join(_SRC, "core"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import board  # noqa: E402  (板边 SSH 编排)
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from PIL import Image  # noqa: E402

# ---- 配置 ----
MODEL_NAME = os.environ.get("HW_MODEL", "model10")
_calib = os.environ.get("HW_CALIB", "")   # 当前校准 json 名(可在 /api/calibrate 后动态更新)
_calib_resolved = False
_calib_last_try = 0.0                      # 失败退避: 板子不可达时 ≥30s 才重试解析
import threading  # noqa
_calib_state = {"status": "idle", "progress": "", "calib": _calib, "error": ""}


def current_calib():
    """返回当前校准名。HW_CALIB 为空时自动向板子询问最新标量校准 json(重启也生效)。
    先看 board.probe_board()(TTL 缓存), 板子不可达不白跑 SSH; 失败后 ≥30s 才重试,
    避免离线时 health 轮询被 latest_calib 的 SSH 超时反复拖慢。"""
    global _calib, _calib_resolved, _calib_last_try
    if not _calib:
        now = time.time()
        if not _calib_resolved and now - _calib_last_try >= 30:
            _calib_last_try = now
            try:
                if board.probe_board()[0]:
                    latest = board.latest_calib()
                    if latest:
                        _calib = latest
                        _calib_state["calib"] = latest
                    _calib_resolved = True   # 成功(或确认无新)才锁定, 不可达则下轮再试
            except Exception:
                pass
        return _calib
    return _calib
CHECK_N = int(os.environ.get("HW_CHECK_N", "100"))   # 判据④ mini-run 默认样本数
WEIGHTS = {"model10": "weights_m10_5400", "model9": "weights_w075ds3"}.get(
    MODEL_NAME, "weights_m10_5400")

CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
           "Industrial", "Pasture", "PermanentCrop", "Residential",
           "River", "SeaLake"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


app = FastAPI(title="决赛真机光计算演示 v3 (pathB)")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def _no_cache(request, call_next):
    resp = await call_next(request)
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.get("/api/health")
def health():
    ok, detail = board.probe_board()
    return {"local": "ok", "remote": ("gazelle-hw:pathB" if ok else "unreachable"),
            "detail": detail, "board": board.BOARD_HOST, "model": MODEL_NAME,
            "weights": WEIGHTS, "calib": current_calib() or "(未校准)",
            "label": "M10 ds3pool3" if MODEL_NAME == "model10" else "M9 w075ds3"}


# ---------------- 放行判据 ----------------

@app.get("/api/checks/canary")
def checks_canary():
    """判据3: MNIST canary — 板端 HW vs numpy 参考 (gap < 0.5pt)。"""
    try:
        r = board.run_mnist(1000, method="dsq")
    except Exception as e:
        return {"name": "③ MNIST canary", "pass": False, "detail": str(e)[:150]}
    if r["acc"] is None or r["ref"] is None:
        return {"name": "③ MNIST canary", "pass": False,
                "value": "acc=%s ref=%s" % (r["acc"], r["ref"]),
                "detail": "板上 runner 未输出参考(检查日志)"}
    ok = r["gap"] < 0.5
    return {"name": "③ MNIST canary gap < 0.5pt", "pass": bool(ok),
            "value": "hw %.2f%% vs ref %.2f%%, gap %.2fpt (n=1000)"% (r["acc"], r["ref"], r["gap"]),
            "detail": "DSQ 三层 MLP, 板上 run_mnist_gazelle 同量化参考"}


@app.get("/api/checks/minirun")
def checks_minirun(n: int = CHECK_N):
    """判据4: EuroSAT mini-run — 板端 acc vs 本地 numpy 干净参考。"""
    n = max(1, min(n, 500))
    x, targets, idx = _eurosat_batch(0, n)
    ref = None
    if x is not None:
        rl = _local_ref(x)
        ref = float(np.mean(np.argmax(rl, 1) == targets)) * 100
    try:
        r = board.run_ds3(0, n, calib_json=current_calib() or None, weights=WEIGHTS)
    except Exception as e:
        return {"name": "④ mini-run 对齐", "pass": False, "detail": str(e)[:150]}
    hw = r["acc"]
    ok = ref is not None and hw is not None and abs(hw - ref) < 2.0
    return {"name": "④ mini-run 采样对齐", "pass": bool(ok),
            "value": "hw %.1f%% vs ref %.1f%%" % (hw if hw is not None else -1,
                                                    ref if ref is not None else -1),
            "detail": "板上 %d 图(路径B) vs 本地 numpy 干净参考, 偏差<2pt" % n}


@app.get("/api/checks/ebr")
def checks_ebr():
    """判据1: EBR 自动测量 — 板上 compass_evb_test, 解析两通道 EBR。"""
    try:
        e1, e2 = board.run_ebr()
    except Exception as ex:
        return {"name": "① EBR ≥ 8", "pass": False, "value": "测量失败",
                "detail": str(ex)[:150]}
    if e1 is None or e2 is None:
        return {"name": "① EBR ≥ 8", "pass": False, "value": "测量失败",
                "detail": "compass_evb_test 未输出 EBR(检查板子/日志)"}
    ok = e1 >= 8 and e2 >= 8
    return {"name": "① EBR ≥ 8", "pass": bool(ok),
            "value": "%.3f / %.3f" % (e1, e2),
            "detail": "板上 compass_evb_test 自动测量(需≥8)"}


# ---------------- 校准 --------------

@app.post("/api/calibrate")
def calibrate():
    """触发重新校准 (compass_cali + 标量), 后台线程, 非阻塞。"""
    if _calib_state["status"] == "running":
        return {"status": "running", "detail": "校准进行中, 请稍候"}
    global _calib
    def work():
        global _calib
        _calib_state["status"] = "running"
        _calib_state["progress"] = "compass_cali (fresh bringup, ~10min)..."
        newname = "calib_scalar_auto_%d.json" % int(time.time())
        res = board.run_calibrate(WEIGHTS, newname)
        _calib_state["progress"] = "scalar_calib 完成" if res["ok"] else res["step"]
        _calib_state["error"] = res.get("err", "")
        if res["ok"]:
            _calib = newname
            _calib_state["calib"] = newname
            _calib_state["status"] = "done"
        else:
            _calib_state["status"] = "error"
    threading.Thread(target=work, daemon=True).start()
    return {"status": "started", "detail": "校准已启动(约10-12分钟)"}


@app.get("/api/calibrate/status")
def calib_status():
    return dict(_calib_state)


# ---------------- 跑批 ----------------

def _eurosat_batch(offset, limit):
    """本地取 EuroSAT 测试集 [offset, offset+limit) 图像+标签 (与板上同 split 对齐)。"""
    data_dir = os.environ.get("HW_DATA",
                              os.path.join(_PKG, "02_复赛_EuroSAT仿真", "data", "EuroSAT_RGB"))
    if not os.path.isdir(data_dir):
        return None, None, None
    sys.path.insert(0, os.path.join(_SRC, "data"))
    from eurosat_split import split_indices  # noqa
    from torchvision import datasets as tvds  # noqa
    ds = tvds.ImageFolder(data_dir)
    _, _, test_idx = split_indices(len(ds), seed=42, val_ratio=0.2, test_ratio=0.2)
    idx = test_idx[offset:offset + limit]
    if len(idx) == 0:
        return None, None, None
    imgs, targets, b64s = [], [], []
    for i in idx:
        path, target = ds.samples[i]
        im = Image.open(path).convert("RGB").resize((64, 64))
        arr = (np.asarray(im, dtype=np.float32) / 255.0 - MEAN) / STD
        imgs.append(arr.transpose(2, 0, 1))
        targets.append(target)
        # 原图缩略图 b64 (用于前端显示)
        import io
        import base64
        buf = io.BytesIO()
        im.save(buf, format="JPEG")
        b64s.append(base64.b64encode(buf.getvalue()).decode())
    return np.stack(imgs).astype(np.float64), np.array(targets), b64s


def _local_ref(x):
    """本地 numpy 干净参考(M10 ds3net) logits。返回 (ws_cache) 复用。"""
    import ds3net  # noqa
    from gazelle_engine import NumpyBackend  # noqa
    weight = os.path.join(_PKG, "03_决赛_EuroSAT真机", "eurosat_research", "weights",
                          "m10_ds3pool3_v8probe15.pth" if MODEL_NAME == "model10"
                          else "m9_j1w075ds3_v8probe15.pth")
    pool = "max3" if MODEL_NAME == "model10" else "max"
    ws, meta = ds3net.load_ds3(weight, pool)
    return ds3net.forward(x, ws, meta, NumpyBackend())


@app.get("/api/run/eurosat")
def run_eurosat(offset: int = 0, limit: int = CHECK_N):
    """路径B 板上跑 ds3 (M10) [offset, offset+limit)。实测 ~3.2s/张。"""
    limit = max(1, min(limit, 200))
    t0 = time.time()
    try:
        r = board.run_ds3(offset, limit, calib_json=current_calib() or None, weights=WEIGHTS)
    except Exception as e:
        raise HTTPException(503, "板上 runner 失败: %s" % e)
    # 本地 numpy 参考用于 gap + 逐图对比 (数据/权重缺失或 offset 越界时
    # 降级为仅板端结果并附 warn, 不再 TypeError -> 500)
    x, targets, b64s = _eurosat_batch(offset, limit)
    warn = ""
    ref_logits = None
    if x is None:
        warn = "本地 EuroSAT 数据缺失或 offset 超出测试集范围, 无逐图对比"
    else:
        try:
            ref_logits = _local_ref(x)
        except Exception as ex:
            warn = "本地参考计算失败(%s), 无逐图对比" % str(ex)[:80]
    ref = None
    if ref_logits is not None:
        ref = float(np.mean(np.argmax(ref_logits, 1) == targets)) * 100
    gap = round(abs(r["acc"] - ref), 2) if r["acc"] is not None and ref is not None else None
    rows = (_build_rows(offset, b64s, targets, r.get("logits"), ref_logits)
            if b64s is not None else [])
    return {"model": MODEL_NAME, "weights": WEIGHTS, "offset": offset, "n": limit,
            "acc": r["acc"], "ref": round(ref, 2) if ref is not None else None,
            "gap": gap, "elapsed_s": r["elapsed_s"],
            "sec_per_img": r["sec_per_img"], "calib": current_calib() or "(none)",
            "engine": "gazelle-hw:pathB", "trace": r["trace"], "rows": rows,
            "warn": warn}


@app.get("/api/mnist/run")
def run_mnist_endpoint(limit: int = 200, offset: int = 0, official: bool = True):
    """MNIST 官方抽样跑批 (路径B)。official=True 用板端 run_mnist_official.py。"""
    limit = max(1, min(limit, 200))
    t0 = time.time()
    try:
        if official:
            r = board.run_mnist(min(limit, 200), method="dsq", official=True)
            r["source"] = "官方 CICC_2026 200 张"
        else:
            r = board.run_mnist(limit, method="dsq", official=False)
            r["source"] = "板上内置官方测试集"
    except Exception as e:
        raise HTTPException(503, "板上 MNIST 失败: %s" % e)
    return {"n": limit, "acc": r["acc"], "ref": r["ref"], "gap": r["gap"],
            "elapsed_s": r["elapsed_s"], "source": r["source"],
            "engine": "gazelle-hw:pathB", "stderr": r["stderr"]}


def _softmax1d(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _topk(z, k=5):
    p = _softmax1d(z)
    idx = np.argsort(-p)[:k]
    return [{"cls": CLASSES[int(i)], "p": round(float(p[i]), 4)} for i in idx]


def _build_rows(offset, b64s, targets, hw_logits, ref_logits):
    """逐图: 图 b64 + 真机 Top5 + 参考 Top5 + 一致性。"""
    rows = []
    n = len(b64s)
    for i in range(n):
        tgt = int(targets[i])
        hw = hw_logits[i] if hw_logits is not None and i < len(hw_logits) else None
        ref = ref_logits[i] if ref_logits is not None else None
        row = {"idx": offset + i, "img": b64s[i], "true": CLASSES[tgt],
               "hw_missing": bool(hw_logits is None or i >= len(hw_logits))}
        if hw is not None:
            row["hw_top1"] = CLASSES[int(np.argmax(hw))]
            row["hw_topk"] = _topk(hw)
            row["hw_correct"] = bool(int(np.argmax(hw)) == tgt)
        if ref is not None:
            row["ref_top1"] = CLASSES[int(np.argmax(ref))]
            row["ref_topk"] = _topk(ref)
            row["ref_correct"] = bool(int(np.argmax(ref)) == tgt)
        if hw is not None and ref is not None:
            row["agree"] = bool(int(np.argmax(hw)) == int(np.argmax(ref)))
        else:
            row["agree"] = None   # 真机或参考缺失, 不误报"不一致"
        rows.append(row)
    return rows


app.mount("/", StaticFiles(directory=os.path.join(_HERE, "web"), html=True),
           name="web")