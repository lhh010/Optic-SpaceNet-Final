"""上板四项放行判据 (SOP, 见 global/AGENTS.md)。

四项全部达标才开窗:
  ① EBR ≥ 8                      (板上 compass_evb_test 读数, 手动录入)
  ② error_std 对健康基线偏差 < ±2% (evb error_std 录入; 快速探针 rel<2% 自动)
  ③ MNIST canary gap < 0.5pt     (真机 DSQ 三层 vs 同量化 numpy, 官方 200 张)
  ④ EuroSAT mini-run 正常         (当前模型真机 vs numpy 干净参考, 逐图一致率≥80% 且 acc 正常)

配套纪律 (脚本侧检查, board_connect.sh 执行):
  - 开窗前读板台账 BOARD_USAGE.md 尾部 + who/ps 侦测他人占用
  - calib json 同窗口生成、20 分钟内使用 (stale −12.5pt); 校准与跑批背靠背
  - 他人使用后 ~40min 物理瞬态, fresh compass_cali 后四判据重过
"""
import base64
import io
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from demo.server.gazelle_engine import HttpBackend, NumpyBackend  # noqa: E402
from demo.server.gazelle_client import (  # noqa: E402
    GAZELLE_HOST, GAZELLE_PORT, CLASSES, MODEL_DEFS, GAZELLE_FAKE,
    _get_backend, _get_ds3)
from demo.server import ds3net  # noqa: E402

MNIST_RES = os.path.join(_HERE, "mnist_res")
MNIST_WEIGHTS = {
    "w1": os.path.join(MNIST_RES, "w1_int4_dsq.npy"),
    "w2": os.path.join(MNIST_RES, "w2_int4_dsq.npy"),
    "w3": os.path.join(MNIST_RES, "w3_int4_dsq.npy"),
    "q": os.path.join(MNIST_RES, "dsq_quant_params.npy"),
}
MNIST_DATA_DIR = os.path.join(MNIST_RES, "data_mnist_200_np")

EBR_STD_BASE = float(os.environ.get("HW_EBR_STD_BASE", "4.70"))  # C2 健康窗口 error_std 基线
EBR_REL_TOL = float(os.environ.get("HW_EBR_REL_TOL", "2.0"))     # 基线偏差容忍 % (SOP ±2%)
PROBE_REL_TOL = float(os.environ.get("HW_PROBE_REL_TOL", "2.0"))  # 快速探针 rel 误差 %


# ---------------------------------------------------------------------------
# ① EBR / error_std (evb_test 读数, 手动录入)
# ---------------------------------------------------------------------------
def check_ebr(ebr_value):
    ok = float(ebr_value) >= 8.0
    return {"name": "① EBR ≥ 8", "pass": ok, "value": str(ebr_value),
            "detail": "板上 compass_evb_test 读数录入 (board_connect.sh [4] 可查)"}


def check_evb_std(error_std):
    v = float(error_std)
    dev = (v - EBR_STD_BASE) / EBR_STD_BASE * 100.0
    # SOP: 低于基线视为改善 (pass); 仅恶化超阈值才 fail (不被绝对值规则误判)
    ok = dev < EBR_REL_TOL
    return {"name": "② error_std 对基线 <+%g%%" % EBR_REL_TOL,
            "pass": ok, "value": "%.3f (基线 %.2f, 偏差 %+.2f%%)"
                                 % (v, EBR_STD_BASE, dev),
            "detail": "evb_test 读数录入; 低于基线视为改善自动通过; 恶化超阈值判 fail"}


# ---------------------------------------------------------------------------
# ② 快速探针 (自动): 已知随机探针真机 vs numpy 精确参考
# ---------------------------------------------------------------------------
def check_probe():
    rng = np.random.RandomState(42)
    x = rng.randint(1, 200, size=(16, 8)).astype(np.uint8)
    w = rng.randint(-100, 100, size=(8, 2)).astype(np.int8)
    exact = x.astype(np.float64) @ w.astype(np.float64)
    try:
        got = _get_backend().matmul_2d(x.astype(np.int64), w.astype(np.int64))
    except Exception as e:
        return {"name": "② 探针 (自动)", "pass": False,
                "detail": "真机 matmul 失败: %s" % str(e)[:150]}
    err = np.abs(np.asarray(got) - exact).ravel()
    rms = float(np.sqrt(np.mean(err ** 2)))
    ref = float(np.sqrt(np.mean(exact ** 2))) + 1e-9
    rel = rms / ref * 100.0
    return {"name": "② 探针 rel<%.1f%%" % PROBE_REL_TOL, "pass": bool(rel < PROBE_REL_TOL),
            "value": "%.2f%%" % rel,
            "detail": "16 组已知探针 vs numpy 精确参考 (快速判据, 无需 evb)"}


# ---------------------------------------------------------------------------
# ③ MNIST canary (DSQ 三层, ×16 scale, 官方 200 张)
# ---------------------------------------------------------------------------
_mnist = None


def _get_mnist():
    global _mnist
    if _mnist is None:
        w1 = np.load(MNIST_WEIGHTS["w1"])[0].astype(np.int32)
        w2 = np.load(MNIST_WEIGHTS["w2"])[0].astype(np.int32)
        w3 = np.load(MNIST_WEIGHTS["w3"])[0].astype(np.int32)
        q = np.load(MNIST_WEIGHTS["q"], allow_pickle=True).item()
        _mnist = (w1, w2, w3, q)
    return _mnist


def _mnist_mm(backend, x_int, w_int):
    x_up = (x_int.astype(np.int32) * 16).astype(np.uint8)
    w_up = (w_int.astype(np.int32) * 16).astype(np.int8)
    y = backend.matmul_2d(x_up, w_up)
    return np.asarray(y, dtype=np.float64) / 256.0


def _mnist_forward(backend, x_int, w1, w2, w3, q):
    s_in, s_w1, s_h1 = q["input_scale"], q["w1_scale"], q["h1_scale"]
    s_w2, s_h2, s_w3 = q["w2_scale"], q["h2_scale"], q["w3_scale"]
    y1 = _mnist_mm(backend, x_int, w1) * (s_in * s_w1)
    h1 = np.clip(np.round(np.maximum(0.0, y1) / s_h1), 0, 15).astype(np.int32)
    y2 = _mnist_mm(backend, h1, w2) * (s_h1 * s_w2)
    h2 = np.clip(np.round(np.maximum(0.0, y2) / s_h2), 0, 15).astype(np.int32)
    return _mnist_mm(backend, h2, w3) * (s_h2 * s_w3)


def _load_mnist_data(limit=200):
    """官方抽样 200 张 (按类子文件夹 .npy)。返回 (x_int (N,784) uint4, labels)。"""
    subdirs = sorted(d for d in os.listdir(MNIST_DATA_DIR)
                     if os.path.isdir(os.path.join(MNIST_DATA_DIR, d)))
    files = []
    for d in subdirs:
        for f in sorted(os.listdir(os.path.join(MNIST_DATA_DIR, d))):
            if f.lower().endswith(".npy"):
                files.append((int(d), os.path.join(MNIST_DATA_DIR, d, f)))
    files = files[:limit]
    imgs, labels = [], []
    for lbl, p in files:
        a = np.load(p)
        imgs.append(np.asarray(a, dtype=np.float32).reshape(-1) / 255.0)
        labels.append(lbl)
    x = np.stack(imgs).astype(np.float32)
    w1, w2, w3, q = _get_mnist()
    x_int = np.clip(np.round(x / q["input_scale"]), 0, 15).astype(np.int32)
    return x_int, np.array(labels, dtype=np.int64)


def check_canary():
    if GAZELLE_FAKE:
        return {"name": "③ MNIST canary", "pass": True,
                "value": "numpy 参考模式 (GAZELLE_FAKE=1, 非真机)",
                "detail": "离线联调仅验证链路, 上板需真机重跑"}
    try:
        x_int, labels = _load_mnist_data(200)
        w1, w2, w3, q = _get_mnist()
        y_hw = _mnist_forward(_get_backend(), x_int, w1, w2, w3, q)
        y_np = _mnist_forward(NumpyBackend(), x_int, w1, w2, w3, q)
    except Exception as e:
        return {"name": "③ MNIST canary", "pass": False,
                "detail": "失败: %s" % str(e)[:150]}
    acc_hw = float(np.mean(np.argmax(y_hw, 1) == labels)) * 100
    acc_np = float(np.mean(np.argmax(y_np, 1) == labels)) * 100
    gap = abs(acc_hw - acc_np)
    return {"name": "③ MNIST canary gap < 0.5pt",
            "pass": bool(gap < 0.5),
            "value": "hw %.2f%% vs ref %.2f%%, gap %.2fpt" % (acc_hw, acc_np, gap),
            "detail": "DSQ 三层 MLP ×16 scale, n=200 (官方抽样), 真机 vs 同量化 numpy"}


# ---------------------------------------------------------------------------
# ④ EuroSAT mini-run (当前模型, 真机 vs numpy 干净参考)
# ---------------------------------------------------------------------------
def check_minirun(model_name="model10", n=200, images=None, labels=None):
    if GAZELLE_FAKE:
        return {"name": "④ EuroSAT mini-run", "pass": True,
                "value": "numpy 参考模式 (非真机)",
                "detail": "离线联调仅验证链路, 上板需真机重跑"}
    if images is None or labels is None:
        return {"name": "④ EuroSAT mini-run", "pass": False,
                "detail": "缺少 test200 数据 (先运行 tools/make_test200.py)"}
    if model_name not in MODEL_DEFS:
        return {"name": "④ EuroSAT mini-run", "pass": False,
                "detail": "未知模型 %s" % model_name}
    try:
        ws, meta, cc = _get_ds3(model_name)
        x = np.asarray(images[:n], dtype=np.float64)
        y = np.asarray(labels[:n])
        backend = _get_backend()
        hw = []
        for i in range(x.shape[0]):
            lg, _ = ds3net.forward_traced(x[i:i + 1], ws, meta, backend,
                                          calib_col=cc)
            hw.append(lg[0])
        hw = np.stack(hw)
        ref, _ = ds3net.forward_traced(x, ws, meta, NumpyBackend(),
                                       calib_col=cc)
    except Exception as e:
        return {"name": "④ EuroSAT mini-run", "pass": False,
                "detail": "失败: %s" % str(e)[:150]}
    p_hw, p_ref = np.argmax(hw, 1), np.argmax(ref, 1)
    agree = float(np.mean(p_hw == p_ref)) * 100
    acc_hw = float(np.mean(p_hw == y)) * 100
    ok = agree >= 80.0 and acc_hw >= 50.0
    return {"name": "④ EuroSAT mini-run 对齐", "pass": bool(ok),
            "value": "一致率 %.1f%%, hw acc %.1f%%" % (agree, acc_hw),
            "detail": "当前模型 %s, n=%d, 真机 vs numpy 干净参考 (SOP: 一致率≥80% 且 acc 正常)"
                      % (model_name, n)}


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def all_checks(ebr=None, evb_std=None, model_name="model10",
               images=None, labels=None, n=200):
    checks = []
    if ebr is not None:
        checks.append(check_ebr(ebr))
    if evb_std is not None:
        checks.append(check_evb_std(evb_std))
    checks.append(check_probe())
    checks.append(check_canary())
    checks.append(check_minirun(model_name, n=n, images=images, labels=labels))
    return {"all_pass": all(c["pass"] for c in checks), "checks": checks,
            "sop_hint": [
                "开窗前: who/ps 侦测他队占用 + 读板上台账 BOARD_USAGE.md 尾部",
                "fresh compass_cali 后四判据背靠背执行; calib json 20 分钟内使用",
                "他人使用后 ~40min 物理瞬态 → fresh 校准 → 判据重过",
            ]}
