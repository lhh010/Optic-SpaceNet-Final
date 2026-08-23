# -*- coding: utf-8 -*-
"""决赛真机演示后端 v2 — 浏览器 × Gazelle 真机光计算（demo-hw）。

v2 变更 (2026-08-23):
  - 模型: M10 ds3pool3 (默认) / M9 w075ds3 — 本地 numpy 前向 (ds3net.py,
    逐行镜像板端 runner 数值语义), 光算 matmul 直连板上 server_gazelle.py
    (不再假设 SSH 隧道, OPTC_HOST 可直接指板 IP)
  - 判据: ② 探针 / ③ 真 MNIST canary (DSQ 三层 MLP) / ④ EuroSAT mini-run
    / ① EBR 手动录入
  - MNIST 官方抽样 200 张跑批 (/api/mnist/run)
  - EuroSAT 分段跑批 (/api/run/eurosat, 默认段 200 张 ≈ 11 min @3.23s/张)

运行 (从 04_决赛演示_真机前端 目录):
  uvicorn demo_hw.server:app --port 8100
环境变量: HW_MODEL=model10|model9  HW_BACKEND=http|numpy
  OPTC_HOST/OPTC_PORT (板上 server_gazelle)  HW_CHUNK(默认2, FPGA行回绕规避)
  HW_CALIB_COL(逐列calib json)  DS3_HEAD_ELEC=1  HW_CHECK_N(默认10)
  HW_MNIST_DIR(官方200张图目录, 内含 png + 可选 labels)
"""
import base64
import io
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
_OPTIC = os.path.join(_PKG, "03_决赛_EuroSAT真机", "opticspacenet")
_MNIST = os.path.join(_PKG, "03_决赛_EuroSAT真机", "mnist")
_SRC = os.path.join(_PKG, "02_复赛_EuroSAT仿真", "src")
for _p in (_OPTIC, _SRC, os.path.join(_SRC, "core"), _HERE, _MNIST):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from PIL import Image  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from typing import Optional  # noqa: E402  (py3.9)

from gazelle_engine import NumpyBackend, HttpBackend  # noqa: E402
import ds3net  # noqa: E402

# ---- 配置(环境变量, 禁位置参数) ----
MODEL_NAME = os.environ.get("HW_MODEL", "model10")          # model10|model9
BACKEND = os.environ.get("HW_BACKEND", "http")              # http|numpy
OPTC_HOST = os.environ.get("OPTC_HOST", "127.0.0.1")
OPTC_PORT = int(os.environ.get("OPTC_PORT", "8000"))
HW_CHUNK = int(os.environ.get("HW_CHUNK", "2"))              # m<=2 tiling (canonical)
CALIB_COL_FILE = os.environ.get("HW_CALIB_COL", "")
HEAD_ELEC = os.environ.get("DS3_HEAD_ELEC", "0") == "1"
CHECK_N = int(os.environ.get("HW_CHECK_N", "10"))
MNIST_DIR = os.environ.get(
    "HW_MNIST_DIR",
    os.path.join(_HERE, "CICC_2026_MNIST_TEST_DATASET", "data_mnist_200_np"))  # 官方 200 张(按类子文件夹 .npy)
PROBE_BASELINE = float(os.environ.get("HW_PROBE_BASELINE", "4.7"))  # 判据②基线(C2健康口径)

_MODELS = {
    "model10": ("m10_ds3pool3_v8probe15.pth", "max3", "M10 ds3pool3"),
    "model9": ("m9_j1w075ds3_v8probe15.pth", "max", "M9 w075ds3"),
}

CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
           "Industrial", "Pasture", "PermanentCrop", "Residential",
           "River", "SeaLake"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((64, 64))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)[None].astype(np.float64)


def get_backend():
    if BACKEND == "http":
        return HttpBackend(host=OPTC_HOST, port=OPTC_PORT,
                            chunk_rows=HW_CHUNK, timeout=120)
    return NumpyBackend()


_ws = None
_meta = None
_calib_col = None


def get_model():
    """返回 (ws, meta, calib_col); 进程级单例, 懒加载."""
    global _ws, _meta, _calib_col
    if _ws is None:
        if MODEL_NAME not in _MODELS:
            raise SystemExit("unknown HW_MODEL %s (want model10/model9)" % MODEL_NAME)
        weight, pool, _label = _MODELS[MODEL_NAME]
        pth = os.path.join(_PKG, "03_决赛_EuroSAT真机", "eurosat_research",
                           "weights", weight)
        _ws, _meta = ds3net.load_ds3(pth, pool)
        _calib_col = ds3net.load_calib_col(CALIB_COL_FILE)
    return _ws, _meta, _calib_col


def infer_batch(x):
    ws, meta, cc = get_model()
    return ds3net.forward(x, ws, meta, get_backend(), calib_col=cc,
                          head_elec=HEAD_ELEC)


# ---------------- MNIST (DSQ 三层 MLP, 初赛 97.35% 链路) ----------------
_mnist = None


def get_mnist():
    """加载 DSQ 权重 (03/mnist/*.npy); 返回 (w1,w2,w3,q,backend)。"""
    global _mnist
    if _mnist is None:
        w1 = np.load(os.path.join(_MNIST, "w1_int4_dsq.npy"))[0].astype(np.int32)
        w2 = np.load(os.path.join(_MNIST, "w2_int4_dsq.npy"))[0].astype(np.int32)
        w3 = np.load(os.path.join(_MNIST, "w3_int4_dsq.npy"))[0].astype(np.int32)
        q = np.load(os.path.join(_MNIST, "dsq_quant_params.npy"),
                    allow_pickle=True).item()
        _mnist = (w1, w2, w3, q)
    return _mnist


def _mnist_mm(backend, x_int, w_int):
    """MNIST scale 模式 (与 run_mnist_gazelle.py 一致): x16 上采后光算再 /256。"""
    x_up = (x_int.astype(np.int32) * 16).astype(np.uint8)
    w_up = (w_int.astype(np.int32) * 16).astype(np.int8)
    y = backend.matmul_2d(x_up, w_up)
    return np.asarray(y, dtype=np.float64) / 256.0


def mnist_forward(backend, x_int, w1, w2, w3, q):
    s_in, s_w1, s_h1 = q["input_scale"], q["w1_scale"], q["h1_scale"]
    s_w2, s_h2, s_w3 = q["w2_scale"], q["h2_scale"], q["w3_scale"]
    y1 = _mnist_mm(backend, x_int, w1) * (s_in * s_w1)
    h1 = np.clip(np.round(np.maximum(0.0, y1) / s_h1), 0, 15).astype(np.int32)
    y2 = _mnist_mm(backend, h1, w2) * (s_h1 * s_w2)
    h2 = np.clip(np.round(np.maximum(0.0, y2) / s_h2), 0, 15).astype(np.int32)
    return _mnist_mm(backend, h2, w3) * (s_h2 * s_w3)


def mnist_np_forward(x_int, w1, w2, w3, q):
    return mnist_forward(NumpyBackend(), x_int, w1, w2, w3, q)


def _mnist_quant(x_float, q):
    return np.clip(np.round(x_float / q["input_scale"]), 0, 15).astype(np.int32)


def load_mnist_images(limit=None, offset=0):
    """官方抽样图 (三类):
    1) HW_MNIST_DIR 为「按类子文件夹 .npy」结构 (CICC_2026_MNIST_TEST_DATASET/data_mnist_200_np/<label>/<i>.npy)
       子文件夹名即真值标签, 每个 .npy 是 (28,28) uint8 (0-255)。
    2) HW_MNIST_DIR 为平铺 png/jpg + 可选 labels.txt。
    3) 回退 03/mnist/test_images.npy (与初赛同源官方测试集)。
    返回 (images (B,784) float[0,1], labels 或 None, names)。"""
    if MNIST_DIR and os.path.isdir(MNIST_DIR):
        # 模式①: 按类子文件夹 .npy
        subdirs = [d for d in os.listdir(MNIST_DIR)
                   if os.path.isdir(os.path.join(MNIST_DIR, d))]
        np_files = []
        for d in sorted(subdirs):
            for f in sorted(os.listdir(os.path.join(MNIST_DIR, d))):
                if f.lower().endswith(".npy"):
                    np_files.append((int(d), os.path.join(MNIST_DIR, d, f)))
        if np_files:
            np_files = np_files[offset:offset + (limit or len(np_files))]
            imgs, labels, names = [], [], []
            for lbl, p in np_files:
                a = np.load(p)  # (28,28) uint8
                imgs.append(np.asarray(a, dtype=np.float32).reshape(-1) / 255.0)
                labels.append(lbl)
                names.append(os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p))
            return np.stack(imgs).astype(np.float32), \
                np.array(labels, dtype=np.int64), names
        # 模式②: 平铺 png/jpg
        names = sorted(f for f in os.listdir(MNIST_DIR)
                       if f.lower().endswith((".png", ".jpg", ".bmp")))
        names = names[offset:offset + (limit or len(names))]
        imgs = []
        for n in names:
            img = Image.open(os.path.join(MNIST_DIR, n)).convert("L")
            if img.size != (28, 28):
                img = img.resize((28, 28))
            imgs.append(np.asarray(img, dtype=np.float32).reshape(-1) / 255.0)
        labels = None
        for lf in ("labels.txt", "labels.csv"):
            p = os.path.join(MNIST_DIR, lf)
            if os.path.exists(p):
                lines = [l.strip() for l in open(p) if l.strip()]
                lab = []
                for l in lines:
                    lab.append(int(l.split(",")[-1]))
                labels = np.array(lab, dtype=np.int64)[offset:offset + len(names)]
                break
        return np.stack(imgs), labels, names
    # 回退: 包内官方测试集前 N 张 (与初赛同源)
    images = np.load(os.path.join(_MNIST, "test_images.npy"))
    labels = np.load(os.path.join(_MNIST, "test_labels.npy"))
    end = min(offset + (limit or len(labels)), len(labels))
    return (images[offset:end].reshape(-1, 784) / 255.0).astype(np.float32), \
        labels[offset:end], ["test_%d" % i for i in range(offset, end)]


app = FastAPI(title="决赛真机光计算演示 v2")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


def _probe_board(timeout=5):
    """可达性探测: 独立短超时 HttpBackend 发一个小矩阵乘。
    注意: 板上 compass_matmul 会按 tia_gain 缩放(交接已证), 返回值非精确 -1;
    健康判定只需端点可达 + 返回有限数值, 不再硬断言 -1。"""
    pb = HttpBackend(host=OPTC_HOST, port=OPTC_PORT, timeout=timeout)
    try:
        x = np.array([[1, 2]], dtype=np.uint8)
        w = np.array([[1], [-1]], dtype=np.int8)
        got = pb.matmul_2d(x, w)
        arr = np.asarray(got, dtype=np.float64).ravel()
        ok = bool(arr.size >= 1 and np.all(np.isfinite(arr)))
        return ok, ("raw=[%s]" % str(arr[:4]))
    except Exception as e:
        return False, str(e)[:150]


@app.get("/api/health")
def health():
    try:
        get_model()
        local = "ok"
    except Exception as e:
        return {"local": "error", "remote": "down", "detail": str(e)[:200]}
    if BACKEND != "http":
        return {"local": local, "remote": "numpy-ref", "model": MODEL_NAME,
                "backend": BACKEND, "label": _MODELS[MODEL_NAME][2]}
    ok, detail = _probe_board()
    return {"local": local,
            "remote": "gazelle-hw:http" if ok else "unreachable",
            "detail": "" if ok else detail,
            "board": "%s:%d" % (OPTC_HOST, OPTC_PORT),
            "model": MODEL_NAME, "backend": BACKEND,
            "label": _MODELS[MODEL_NAME][2], "chunk": HW_CHUNK}


class InferRequest(BaseModel):
    image_b64: str
    label: Optional[str] = None


@app.post("/api/infer")
def infer(req: InferRequest):
    try:
        x = preprocess(base64.b64decode(req.image_b64))
    except Exception as e:
        raise HTTPException(400, "image decode failed: %s" % e)
    t0 = time.perf_counter()
    try:
        logits = infer_batch(x)
        probs = np.exp(logits[0] - logits[0].max())
        probs = probs / probs.sum()
    except Exception as e:
        raise HTTPException(503, "真机推理失败(检查板上 server/校准): %s" % e)
    latency = time.perf_counter() - t0
    order = np.argsort(-probs)[:5]
    top = int(order[0])
    return {"pred": CLASSES[top],
            "topk": [{"cls": CLASSES[i], "p": round(float(probs[i]), 4)}
                      for i in order],
            "latency_s": round(latency, 3),
            "engine": "gazelle-hw" if BACKEND == "http" else "numpy-ref",
            "model": MODEL_NAME,
            "correct": (req.label == CLASSES[top]) if req.label else None}


# ---------------- 放行判据 (四项) ----------------

@app.get("/api/checks/probe")
def checks_probe():
    """判据2: 探针矩阵乘误差 (vs numpy 精确参考)。"""
    rng = np.random.RandomState(42)
    x = rng.randint(1, 200, size=(16, 8)).astype(np.uint8)
    w = rng.randint(-100, 100, size=(8, 2)).astype(np.int8)
    exact = x.astype(np.float64) @ w.astype(np.float64)
    try:
        got = get_backend().matmul_2d(x.astype(np.int64), w.astype(np.int64))
    except Exception as e:
        return {"name": "② error_std 偏差", "pass": False,
                "detail": "真机 matmul 失败: %s" % str(e)[:150]}
    err = np.abs(got - exact).ravel()
    rms = float(np.sqrt(np.mean(err ** 2)))
    ref = float(np.sqrt(np.mean(np.abs(exact) ** 2))) + 1e-9
    rel = rms / ref * 100.0
    return {"name": "② error_std 偏差 < ±2%", "pass": bool(rel < 2.0),
            "value": "%.2f%%" % rel,
            "detail": "16 组已知探针 vs numpy 精确参考, 相对 RMS 误差"}


@app.get("/api/checks/canary")
def checks_canary():
    """判据3: 真 MNIST canary — HW vs numpy 同量化参考 (gap < 0.5pt)。"""
    w1, w2, w3, q = get_mnist()
    x, labels, _ = load_mnist_images(limit=200)
    if labels is None:
        return {"name": "③ MNIST canary", "pass": False,
                "detail": "无标签, canary 需要标签 (labels.txt 或回退 npy)"}
    x_int = _mnist_quant(x, q)
    try:
        y_hw = mnist_forward(get_backend(), x_int, w1, w2, w3, q)
    except Exception as e:
        return {"name": "③ MNIST canary", "pass": False,
                "detail": "真机失败: %s" % str(e)[:150]}
    y_np = mnist_np_forward(x_int, w1, w2, w3, q)
    acc_hw = float(np.mean(np.argmax(y_hw, 1) == labels)) * 100
    acc_np = float(np.mean(np.argmax(y_np, 1) == labels)) * 100
    gap = abs(acc_hw - acc_np)
    return {"name": "③ MNIST canary gap < 0.5pt", "pass": bool(gap < 0.5),
            "value": "hw %.2f%% vs ref %.2f%%, gap %.2fpt" % (acc_hw, acc_np, gap),
            "detail": "DSQ 三层 MLP, n=%d, 真机 vs 同量化 numpy 参考" % len(labels)}


@app.get("/api/checks/minirun")
def checks_minirun():
    """判据4: EuroSAT mini-run — M10 hw vs numpy 干净参考 逐图对齐。"""
    data_dir = os.environ.get(
        "HW_DATA",
        os.path.join(_PKG, "02_复赛_EuroSAT仿真", "data", "EuroSAT_RGB"))
    if not os.path.isdir(data_dir):
        raise HTTPException(503, "EuroSAT 数据未找到: %s" % data_dir)
    sys.path.insert(0, os.path.join(_SRC, "data"))
    from eurosat_split import split_indices  # noqa
    from torchvision import datasets as tvds  # noqa
    ds = tvds.ImageFolder(data_dir)
    _, _, test_idx = split_indices(len(ds), seed=42, val_ratio=0.2, test_ratio=0.2)
    rng = np.random.RandomState(7)
    picks = sorted(rng.choice(test_idx, size=min(CHECK_N, len(test_idx)),
                              replace=False).tolist())
    imgs, targets = [], []
    for i in picks:
        path, target = ds.samples[i]
        im = Image.open(path).convert("RGB").resize((64, 64))
        arr = (np.asarray(im, dtype=np.float32) / 255.0 - MEAN) / STD
        imgs.append(arr.transpose(2, 0, 1))
        targets.append(target)
    x = np.stack(imgs).astype(np.float64)
    try:
        y_hw = infer_batch(x)
    except Exception as e:
        return {"name": "④ mini-run 对齐", "pass": False,
                "detail": "真机失败: %s" % str(e)[:150]}
    y_np = ds3net.forward(x, *get_model()[:2], NumpyBackend(),
                          calib_col={}, head_elec=HEAD_ELEC)
    p_hw = np.argmax(y_hw, 1)
    p_np = np.argmax(y_np, 1)
    agree = int(np.sum(p_hw == p_np))
    acc_hw = float(np.mean(p_hw == np.array(targets))) * 100
    return {"name": "④ mini-run 采样对齐",
            "pass": bool(agree / len(picks) >= 0.8 and acc_hw >= 50),
            "value": "%d/%d 一致, hw acc %.1f%%" % (agree, len(picks), acc_hw),
            "detail": "逐图预测一致率 + 精度正常性 (n=%d)" % len(picks)}


@app.post("/api/checks/ebr")
def checks_ebr(req: dict):
    """判据1: EBR 手动录入 (板端 compass_evb_test 读数)。"""
    try:
        v = float(req.get("ebr"))
    except (TypeError, ValueError):
        raise HTTPException(400, "ebr 字段缺失/非数值")
    return {"name": "① EBR ≥ 8", "pass": bool(v >= 8), "value": str(v),
            "detail": "板端 compass_evb_test 实测值(现场读数录入)"}


# ---------------- 跑批段 (30min 窗口预算: 校准10 + 判据5 + 跑批15) ----------------

@app.get("/api/run/eurosat")
def run_eurosat(offset: int = 0, limit: int = 200):
    """EuroSAT 分段跑批 [offset, offset+limit), 全量 5400 按 200/段。
    实测吞吐 (C2): M10 3.23s/张, M9 1.98s/张 (板端 runner 口径);
    HTTP 直连 m<=2 tiling 会更慢, 首次实测后调整 limit。"""
    limit = max(1, min(limit, 500))
    data_dir = os.environ.get(
        "HW_DATA",
        os.path.join(_PKG, "02_复赛_EuroSAT仿真", "data", "EuroSAT_RGB"))
    if not os.path.isdir(data_dir):
        raise HTTPException(503, "EuroSAT 数据未找到: %s" % data_dir)
    sys.path.insert(0, os.path.join(_SRC, "data"))
    from eurosat_split import split_indices  # noqa
    from torchvision import datasets as tvds  # noqa
    ds = tvds.ImageFolder(data_dir)
    _, _, test_idx = split_indices(len(ds), seed=42, val_ratio=0.2, test_ratio=0.2)
    idx = test_idx[offset:offset + limit]
    if len(idx) == 0:
        raise HTTPException(400, "offset %d 超出测试集" % offset)
    backend = get_backend()
    ws, meta, cc = get_model()
    correct = 0
    t0 = time.perf_counter()
    per = []
    B = 8
    for s in range(0, len(idx), B):
        batch_idx = idx[s:s + B]
        imgs, targets = [], []
        for i in batch_idx:
            path, target = ds.samples[i]
            im = Image.open(path).convert("RGB").resize((64, 64))
            arr = (np.asarray(im, dtype=np.float32) / 255.0 - MEAN) / STD
            imgs.append(arr.transpose(2, 0, 1))
            targets.append(target)
        try:
            logits = ds3net.forward(np.stack(imgs).astype(np.float64),
                                    ws, meta, backend, calib_col=cc,
                                    head_elec=HEAD_ELEC)
        except Exception as e:
            raise HTTPException(503, "真机失败(第 %d 张起): %s" % (s, e))
        preds = np.argmax(logits, 1)
        correct += int(np.sum(preds == np.array(targets)))
        done = s + len(batch_idx)
        per.append({"n": done, "acc": round(correct * 100.0 / done, 2),
                    "elapsed": round(time.perf_counter() - t0, 1)})
    n = len(idx)
    return {"model": MODEL_NAME, "offset": offset, "n": n,
            "acc": round(correct * 100.0 / n, 2),
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "sec_per_img": round((time.perf_counter() - t0) / n, 2),
            "engine": "gazelle-hw" if BACKEND == "http" else "numpy-ref",
            "trace": per}


@app.get("/api/mnist/run")
def run_mnist(limit: int = 200, offset: int = 0):
    """MNIST 官方抽样跑批 (默认 200 张, DSQ 三层 MLP)。"""
    limit = max(1, min(limit, 1000))
    x, labels, names = load_mnist_images(limit=limit, offset=offset)
    w1, w2, w3, q = get_mnist()
    x_int = _mnist_quant(x, q)
    t0 = time.perf_counter()
    try:
        y = mnist_forward(get_backend(), x_int, w1, w2, w3, q)
    except Exception as e:
        raise HTTPException(503, "真机失败: %s" % e)
    preds = np.argmax(y, 1)
    out = {"n": len(preds), "elapsed_s": round(time.perf_counter() - t0, 1),
           "engine": "gazelle-hw" if BACKEND == "http" else "numpy-ref",
           "source": MNIST_DIR if MNIST_DIR else "内置官方测试集"}
    if labels is not None:
        acc = float(np.mean(preds == labels)) * 100
        out["acc"] = round(acc, 2)
        wrong = [names[i] for i in range(len(preds))
                 if preds[i] != labels[i]][:20]
        out["wrong_sample"] = wrong
    else:
        out["pred_head"] = [int(p) for p in preds[:20]]
        out["note"] = "无标签文件, 只返回预测"
    return out


app.mount("/", StaticFiles(directory=os.path.join(_HERE, "web"), html=True),
           name="web")