"""Gazelle 真机客户端 — 替代 remote_client(容器 osimulator 路径)。

接口与 remote_client 完全一致 (health/infer/RemoteUnavailable), 因此
app.py 只需把 import 换成 gazelle_client, 前端 (demo/web) 零改动:

  GET  /api/health  → 真机探针 (HttpBackend 小 matmul) → {"status","engine"}
  POST /api/infer   → 光电分离: 本地 torch 电计算 (stem/BN/ReLU/Pool) +
                      光层 matmul 经 HttpBackend → 板上 server_gazelle.py
                      → compass_sdk → Gazelle 光芯片

数据流:
  浏览器 ── /api/infer ──▶ gazelle_client ── HTTP :8000 ──▶ 板上 server_gazelle.py
                              │ (torch 电计算, model_trace 同源 Model 3)
                              ▼
                       PathResult (engine="gazelle-osim", 与 optic_server 同构,
                       含逐层激活 act_b64 → 前端光|电逐层对比)

降级链: 任何失败 raise RemoteUnavailable → app.py 回退本地 fake 引擎
(meta.degraded=true, 前端黄色提示) —— 与旧路径行为一致。

连接: 板上 `ssh uisrc@192.168.31.158` (密码 5182) 运行
      `server_gazelle.py` (:8000); 本机无需装 compass_sdk/osimulator。

环境变量:
  GAZELLE_HOST    板上 IP (默认 192.168.31.158)
  GAZELLE_PORT    板上服务端口 (默认 8000)
  GAZELLE_WEIGHT  Model 3 权重 (默认 <repo>/weights/spacenet_v2_phase4_v3_int8.pth)
  GAZELLE_CALIB   可选逐通道修正 .npz (analyze_layers 产物, 键=权重md5)
  GAZELLE_FAKE=1  离线 numpy 参考 (不占板, 联调用)
  GAZELLE_TIMEOUT 单次 HTTP 超时秒 (默认 300)
"""
import base64
import io
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))           # demo/server
REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))           # repo root
for _p in (_HERE, os.path.join(REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# optic_layers (src/core) 由 app.py 的 _pathsetup 注入; 直接 import 兜底:
_CORE = os.path.join(REPO_ROOT, "src", "core")
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from demo.server import model_trace  # noqa: E402
from demo.server.inference_local import preprocess  # noqa: E402
from demo.server.gazelle_engine import (  # noqa: E402
    HttpBackend, NumpyBackend, GazelleOpticalEngine)
from optic_layers import build_optical_model  # noqa: E402

# ---------------------------------------------------------------------------
# 配置 (禁位置参数)
# ---------------------------------------------------------------------------
GAZELLE_HOST = os.environ.get("GAZELLE_HOST", "192.168.31.158")
GAZELLE_PORT = int(os.environ.get("GAZELLE_PORT", "8000"))
GAZELLE_WEIGHT = os.environ.get(
    "GAZELLE_WEIGHT",
    os.path.join(REPO_ROOT, "weights", "spacenet_v2_phase4_v3_int8.pth"))
GAZELLE_CALIB = os.environ.get("GAZELLE_CALIB", "")
GAZELLE_FAKE = os.environ.get("GAZELLE_FAKE", "0") == "1"
GAZELLE_TIMEOUT = float(os.environ.get("GAZELLE_TIMEOUT", "300"))

ENGINE_LABEL = "gazelle-osim"  # 前端 health 状态灯已认识该标签 (青)

CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]


class RemoteUnavailable(Exception):
    """Gazelle 真机不可用 (app.py 捕获后降级本地 fake 引擎)。"""


# ---------------------------------------------------------------------------
# 模型单例 (惰性加载)
# ---------------------------------------------------------------------------
_state = {"model": None, "engine": None}


def _load_correction(path):
    """analyze_layers 校准 npz → {weight_md5: (a_j, b_j)}。"""
    z = np.load(path)
    correction = {}
    for k in z.files:
        arr = z[k]  # (2, n) -> a_j, b_j
        correction[k] = (np.asarray(arr[0], dtype=np.float64),
                         np.asarray(arr[1], dtype=np.float64))
    return correction


def _get_backend():
    if GAZELLE_FAKE:
        return NumpyBackend()
    return HttpBackend(host=GAZELLE_HOST, port=GAZELLE_PORT,
                       timeout=GAZELLE_TIMEOUT)


def _get_model():
    """返回 (optical_model, engine); 进程级单例, 懒加载。"""
    if _state["model"] is None:
        if not os.path.isfile(GAZELLE_WEIGHT):
            raise RemoteUnavailable("weight not found: %s" % GAZELLE_WEIGHT)
        backend = _get_backend()
        correction = None
        if GAZELLE_CALIB and os.path.isfile(GAZELLE_CALIB):
            correction = _load_correction(GAZELLE_CALIB)
        engine = GazelleOpticalEngine(backend, correction=correction)
        model = model_trace.build_student(GAZELLE_WEIGHT)
        # stem(1×1 3→8) 电计算; stage1/2/3 + fc1/fc2 光计算 (int8)
        build_optical_model(model, engine, pad_to_8=True,
                            input_bit=8, weight_bit=8,
                            keep_first_conv_electronic=True,
                            convert_linear=True)
        model.eval()
        _state["model"] = model
        _state["engine"] = engine
    return _state["model"], _state["engine"]


# ---------------------------------------------------------------------------
# 接口: health / infer (与 remote_client 同签名)
# ---------------------------------------------------------------------------
def health(base_url=None):
    """探针: 板上服务可达 + 返回有限数值 → ok。不可用 raise RemoteUnavailable。"""
    if GAZELLE_FAKE:
        return {"status": "ok", "engine": ENGINE_LABEL,
                "detail": "numpy 离线参考 (GAZELLE_FAKE=1)"}
    try:
        pb = HttpBackend(host=GAZELLE_HOST, port=GAZELLE_PORT, timeout=5)
        x = np.array([[1, 2]], dtype=np.uint8)
        w = np.array([[1], [-1]], dtype=np.int8)
        got = np.asarray(pb.matmul_2d(x, w), dtype=np.float64).ravel()
        if not (got.size >= 1 and np.all(np.isfinite(got))):
            raise RemoteUnavailable("probe 返回非有限值: %s" % got)
    except RemoteUnavailable:
        raise
    except Exception as e:
        raise RemoteUnavailable("probe failed: %s" % str(e)[:150])
    return {"status": "ok", "engine": ENGINE_LABEL,
            "detail": "raw=[%s]" % str(got[:4])}


def _encode_act(act_tensor):
    arr = act_tensor.detach().cpu().numpy()[0].astype(np.float16)
    buf = io.BytesIO()
    np.savez(buf, act=arr)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def infer(image_b64, base_url=None, model_id=None):
    """POST 语义: 本地电计算 + 真机光计算 → PathResult (optic_server 同构)。

    当前真机链路只挂载 Model 3 (主演示页); compare 页面的 Model 1/2
    请求将 raise RemoteUnavailable → compare_models 自动降级本地 fake。"""
    if model_id not in (None, 3):
        raise RemoteUnavailable(
            "真机链路仅挂载 Model 3; Model %s 走本地 fake 降级" % model_id)
    model, engine = _get_model()
    try:
        image_bytes = base64.b64decode(image_b64)
        img_tensor = preprocess(image_bytes)
    except Exception as e:
        raise RemoteUnavailable("image decode failed: %s" % e)

    t0 = time.perf_counter()
    try:
        traced = model_trace.forward_traced(model, img_tensor)
    except Exception as e:
        raise RemoteUnavailable("真机推理失败: %s" % str(e)[:200])

    logits = traced["logits"][0]
    probs = torch.softmax(logits, dim=0)
    order = torch.argsort(probs, descending=True)
    prob_dict = {CLASSES[i]: round(float(probs[i]), 6) for i in order.tolist()}

    layers = [{
        "name": l["name"],
        "where": l["where"],
        "spec": l["spec"],
        "shape": list(l["act"].shape[1:]),
        "latency_s": round(l["latency_s"], 6),
        "act_b64": _encode_act(l["act"]),
    } for l in traced["layers"]]

    return {
        "engine": ENGINE_LABEL,
        "model_id": 3,
        "pred": CLASSES[int(logits.argmax())],
        "probs": prob_dict,
        "latency_total_s": round(sum(l["latency_s"] for l in layers), 6),
        "layers": layers,
    }
