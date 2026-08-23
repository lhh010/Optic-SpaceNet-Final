"""Gazelle 真机客户端 — 替代 remote_client(容器 osimulator 路径), 支持多模型。

接口与 remote_client 一致 (health/infer/RemoteUnavailable), 因此 app.py 只需
把 import 换成 gazelle_client, 前端 (demo/web) 仅新增模型选择下拉。

支持模型 (model_name):
  model3   SpaceNet V2 + KD (复赛主线, torch 前向 + optic_layers 光层替换)
  model9   M9 w075ds3 (≤2M 预算冠军, 真机全量 94.43%)
  model10  M10 ds3pool3 (SOTA, 真机全量 95.33%)   ← numpy 前向 (ds3net.py,
           逐行镜像 run_ds3_gazelle.py, 与板端 canonical 链路一致)

数据流 (光电分离):
  浏览器 ── /api/infer ──▶ gazelle_client ── HTTP :8000 ──▶ 板上 server_gazelle.py
                              │ (电计算: stem/BN/ReLU/GAP; 光层 matmul 走 HTTP)
                              ▼
                       PathResult (engine="gazelle-osim", 逐层激活 act_b64)

降级链: 任何失败 raise RemoteUnavailable → app.py 回退本地 fake/干净参考。

连接: 板上 `ssh uisrc@192.168.31.158` (密码 5182) 运行 `server_gazelle.py` (:8000)。

环境变量:
  GAZELLE_HOST       板上 IP (默认 192.168.31.158)
  GAZELLE_PORT       板上服务端口 (默认 8000)
  GAZELLE_WEIGHT     Model 3 权重 (默认 <repo>/weights/spacenet_v2_phase4_v3_int8.pth)
  GAZELLE_WEIGHT_9   M9 权重 (默认 <eurosat_research>/weights/m9_j1w075ds3_v8probe15.pth)
  GAZELLE_WEIGHT_10  M10 权重 (默认 <eurosat_research>/weights/m10_ds3pool3_v8probe15.pth)
  GAZELLE_CALIB      Model 3 逐通道修正 npz (analyze_layers 产物)
  GAZELLE_CALIB_9/10 M9/M10 逐列 calib json (calibrate_col.py 产物, 同窗口)
  GAZELLE_FAKE=1     离线 numpy 参考 (不占板, 联调用)
  GAZELLE_TIMEOUT    单次 HTTP 超时秒 (默认 300)
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
_CORE = os.path.join(REPO_ROOT, "src", "core")
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from demo.server import model_trace  # noqa: E402
from demo.server.inference_local import preprocess  # noqa: E402
from demo.server.gazelle_engine import (  # noqa: E402
    HttpBackend, NumpyBackend, GazelleOpticalEngine)
from demo.server.backend_transports import (  # noqa: E402
    TransportUnavailable, matrix_backend)
from demo.server import ds3net  # noqa: E402
from optic_layers import build_optical_model  # noqa: E402

# ---------------------------------------------------------------------------
# 配置 (禁位置参数)
# ---------------------------------------------------------------------------
GAZELLE_HOST = os.environ.get("GAZELLE_HOST", "192.168.31.158")
GAZELLE_PORT = int(os.environ.get("GAZELLE_PORT", "8000"))
_ER_WEIGHTS = os.path.join(os.path.dirname(REPO_ROOT), "osim",
                           "eurosat_research", "weights")
GAZELLE_WEIGHT = os.environ.get(
    "GAZELLE_WEIGHT",
    os.path.join(REPO_ROOT, "weights", "spacenet_v2_phase4_v3_int8.pth"))
GAZELLE_WEIGHT_9 = os.environ.get(
    "GAZELLE_WEIGHT_9",
    os.path.join(_ER_WEIGHTS, "m9_j1w075ds3_v8probe15.pth"))
GAZELLE_WEIGHT_10 = os.environ.get(
    "GAZELLE_WEIGHT_10",
    os.path.join(_ER_WEIGHTS, "m10_ds3pool3_v8probe15.pth"))
GAZELLE_CALIB = os.environ.get("GAZELLE_CALIB", "")
GAZELLE_CALIB_9 = os.environ.get("GAZELLE_CALIB_9", "")
GAZELLE_CALIB_10 = os.environ.get("GAZELLE_CALIB_10", "")
GAZELLE_FAKE = os.environ.get("GAZELLE_FAKE", "0") == "1"
GAZELLE_TIMEOUT = float(os.environ.get("GAZELLE_TIMEOUT", "300"))

ENGINE_LABEL = "gazelle-hardware"  # 与 Model 3 的 gazelle-osim 明确区分

CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

MODEL_DEFS = {
    "model3": {"label": "Model 3 · SpaceNet V2 + KD",
               "weight": GAZELLE_WEIGHT, "calib": GAZELLE_CALIB,
               "stem_pool": None, "head_elec": False},
    "model9": {"label": "M9 · w075ds3 (1.52M MACs)",
               "weight": GAZELLE_WEIGHT_9, "calib": GAZELLE_CALIB_9,
               "stem_pool": "max", "head_elec": False},
    "model10": {"label": "M10 · ds3pool3 (2.56M MACs)",
                "weight": GAZELLE_WEIGHT_10, "calib": GAZELLE_CALIB_10,
                "stem_pool": "max3", "head_elec": False},
}


class RemoteUnavailable(Exception):
    """Gazelle 真机不可用 (app.py 捕获后降级本地 fake 引擎)。"""


# ---------------------------------------------------------------------------
# 模型单例 (惰性加载, 按 model_name)
# ---------------------------------------------------------------------------
_state = {"model3": {}, "model9": None, "model10": None,
         "model3_clean": None}


def _load_correction(path):
    """analyze_layers 校准 npz → {weight_md5: (a_j, b_j)}。"""
    z = np.load(path)
    correction = {}
    for k in z.files:
        arr = z[k]  # (2, n) -> a_j, b_j
        correction[k] = (np.asarray(arr[0], dtype=np.float64),
                         np.asarray(arr[1], dtype=np.float64))
    return correction


def _get_backend(backend_mode=None):
    if backend_mode in ("osimulator", "gazelle_ssh", "gazelle_serial"):
        try:
            return matrix_backend(backend_mode)
        except TransportUnavailable as exc:
            raise RemoteUnavailable(str(exc))
    if GAZELLE_FAKE:
        return NumpyBackend()
    backend = HttpBackend(host=GAZELLE_HOST, port=GAZELLE_PORT,
                          timeout=GAZELLE_TIMEOUT)
    backend.name = ENGINE_LABEL
    return backend


def _get_model3(backend_mode=None):
    """Model 3 torch forward with one cached optical model per transport."""
    key = backend_mode or "gazelle_http"
    if key not in _state["model3"]:
        if not os.path.isfile(GAZELLE_WEIGHT):
            raise RemoteUnavailable("weight not found: %s" % GAZELLE_WEIGHT)
        correction = None
        if GAZELLE_CALIB and os.path.isfile(GAZELLE_CALIB):
            correction = _load_correction(GAZELLE_CALIB)
        engine = GazelleOpticalEngine(
            _get_backend(backend_mode), correction=correction)
        model = model_trace.build_student(GAZELLE_WEIGHT)
        build_optical_model(model, engine, pad_to_8=True,
                            input_bit=8, weight_bit=8,
                            keep_first_conv_electronic=True,
                            convert_linear=True)
        model.eval()
        _state["model3"][key] = (model, engine)
    return _state["model3"][key]


def _get_ds3(model_name):
    """M9/M10: ds3net numpy 前向 (镜像 run_ds3_gazelle.py)。"""
    if _state[model_name] is None:
        d = MODEL_DEFS[model_name]
        if not os.path.isfile(d["weight"]):
            raise RemoteUnavailable("weight not found: %s" % d["weight"])
        ws, meta = ds3net.load_ds3(d["weight"], d["stem_pool"])
        cc = ds3net.load_calib_col(d["calib"])
        _state[model_name] = (ws, meta, cc)
    return _state[model_name]


def _get_model(model_name):
    if model_name == "model3":
        return _get_model3()
    return _get_ds3(model_name)


# ---------------------------------------------------------------------------
# 接口: health / infer (与 remote_client 同签名 + model 选择)
# ---------------------------------------------------------------------------
def health(base_url=None, backend_mode=None):
    """Probe the selected matrix transport with a finite-value matmul."""
    try:
        pb = _get_backend(backend_mode)
        x = np.array([[1, 2]], dtype=np.uint8)
        w = np.array([[1], [-1]], dtype=np.int8)
        got = np.asarray(pb.matmul_2d(x, w), dtype=np.float64).ravel()
        if not (got.size >= 1 and np.all(np.isfinite(got))):
            raise RemoteUnavailable("probe 返回非有限值: %s" % got)
    except RemoteUnavailable:
        raise
    except Exception as exc:
        raise RemoteUnavailable("probe failed: %s" % str(exc)[:180])
    label = getattr(pb, "name", ENGINE_LABEL)
    if label == "http":
        label = ENGINE_LABEL
    return {"status": "ok", "engine": label,
            "detail": "raw=[%s]" % str(got[:4])}


def _encode_act(act):
    """act: torch.Tensor 或 np.ndarray → act_b64 (float16 npz, 与 optic_server 一致)。"""
    if act is None:
        return None
    if torch.is_tensor(act):
        arr = act.detach().cpu().numpy()[0].astype(np.float16)
    else:
        arr = np.asarray(act, dtype=np.float32)[0].astype(np.float16)
    buf = io.BytesIO()
    np.savez(buf, act=arr)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _path_result(logits, layers, model_name, engine_label, clean=False):
    """组装 PathResult；真机与本地 clean reference 使用不同引擎标签。"""
    lg = logits[0] if torch.is_tensor(logits) else logits[0]
    if torch.is_tensor(lg):
        probs_t = torch.softmax(lg, dim=0).tolist()
        pred_i = int(lg.argmax())
    else:
        e = np.exp(lg - lg.max())
        probs_t = (e / e.sum()).tolist()
        pred_i = int(np.argmax(lg))
    order = np.argsort(-np.asarray(probs_t))
    prob_dict = {CLASSES[i]: round(float(probs_t[i]), 6) for i in order.tolist()}
    out_layers = [{
        "name": l["name"], "where": l["where"], "spec": l["spec"],
        "analysis": l.get("analysis"),
        "shape": l.get("shape") or list(l["act"].shape[1:]),
        "latency_s": round(l.get("latency_s", 0.0), 6),
        "act_b64": _encode_act(l["act"]),
    } for l in layers]
    return {
        "engine": engine_label,
        "model": model_name,
        "pred": CLASSES[pred_i],
        "probs": prob_dict,
        "latency_total_s": round(sum(l.get("latency_s", 0.0) for l in layers), 6),
        "layers": out_layers,
        "clean": clean,
    }


def infer(image_b64, base_url=None, model_id=None, model_name="model3",
          clean=False, backend_mode=None):
    """Quantized optical/hardware inference through the selected transport."""
    if model_id is not None and model_name == "model3":
        if model_id == 9:
            model_name = "model9"
        elif model_id == 10:
            model_name = "model10"
        elif model_id != 3:
            raise RemoteUnavailable("未知 model_id %s (支持 3/9/10)" % model_id)
    if model_name not in MODEL_DEFS:
        raise RemoteUnavailable("未知模型 %s (支持 model3/model9/model10)" % model_name)

    try:
        image_bytes = base64.b64decode(image_b64)
        img_tensor = preprocess(image_bytes)
    except Exception as exc:
        raise RemoteUnavailable("image decode failed: %s" % exc)

    t0 = time.perf_counter()
    engine_label = "numpy-clean"
    try:
        if model_name == "model3":
            if clean:
                if _state["model3_clean"] is None:
                    eng = GazelleOpticalEngine(NumpyBackend())
                    model = model_trace.build_student(
                        MODEL_DEFS["model3"]["weight"])
                    build_optical_model(
                        model, eng, pad_to_8=True, input_bit=8,
                        weight_bit=8, keep_first_conv_electronic=True,
                        convert_linear=True)
                    model.eval()
                    _state["model3_clean"] = model
                traced = model_trace.forward_traced(
                    _state["model3_clean"], img_tensor)
                logits, layers = traced["logits"], traced["layers"]
            else:
                model, engine = _get_model3(backend_mode)
                traced = model_trace.forward_traced(model, img_tensor)
                logits, layers = traced["logits"], traced["layers"]
                engine_label = getattr(engine.backend, "name", ENGINE_LABEL)
        else:
            ws, meta, cc = _get_ds3(model_name)
            matrix = NumpyBackend() if clean else _get_backend(backend_mode)
            x = np.asarray(img_tensor.numpy(), dtype=np.float64)
            logits, layers = ds3net.forward_traced(
                x, ws, meta, matrix, calib_col=cc,
                head_elec=MODEL_DEFS[model_name]["head_elec"])
            if not clean:
                engine_label = getattr(matrix, "name", ENGINE_LABEL)
    except RemoteUnavailable:
        raise
    except TransportUnavailable as exc:
        raise RemoteUnavailable(str(exc))
    except Exception as exc:
        target = backend_mode or "Gazelle"
        raise RemoteUnavailable(
            "%s 推理失败: %s" % (target, str(exc)[:220]))

    if engine_label == "http":
        engine_label = ENGINE_LABEL
    result = _path_result(
        logits, layers, model_name, engine_label, clean=clean)
    result["latency_total_s"] = round(
        max(result["latency_total_s"], time.perf_counter() - t0), 6)
    return result


def infer_fp32(image_b64, model_name="model3"):
    """Run the selected model with original floating-point weights locally."""
    try:
        image_bytes = base64.b64decode(image_b64)
        img_tensor = preprocess(image_bytes)
        if model_name == "model3":
            from demo.server import inference_local
            return inference_local.infer_fp32(img_tensor)
        ws, meta, _ = _get_ds3(model_name)
        x = np.asarray(img_tensor.numpy(), dtype=np.float64)
        logits, layers = ds3net.forward_fp32_traced(x, ws, meta)
        return _path_result(
            logits, layers, model_name, "numpy-fp32", clean=True)
    except RemoteUnavailable:
        raise
    except Exception as exc:
        raise RemoteUnavailable(
            "本地 FP32 %s 推理失败: %s" %
            (model_name, str(exc)[:220]))
