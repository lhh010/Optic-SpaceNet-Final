# -*- coding: utf-8 -*-
"""GazelleOpticalEngine — OpticalEngine subclass that sends optical matmuls to
the REAL Gazelle board over HTTP (or to exact numpy as a clean reference).

Reuses every piece of the baseline 90.43% inference path from
train-test/src/core/optic_layers.py (OpticConv2d/OpticLinear/build_optical_model)
and only replaces the two osimulator call sites (_matmul_real,
matmul_pre_quantized) with a 2-D backend (compass semantics: (m,k)@(k,n)).

Hardware facts used here (measured on the board):
  * compass_matmul(vec_uint8, w_int8) returns ~ (vec @ w) in integer MAC units
    directly (the SDK already multiplies the 12-bit ADC readout by tia_gain).
  * Shape limit: keep m <= 1024 per call (chunked by the backend).
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn

import optic_layers as OL

_HERE = os.path.dirname(os.path.abspath(__file__))
_TS = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, "train-test"))

# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class NumpyBackend(object):
    """Exact integer matmul — clean reference (no hardware error)."""

    name = "numpy"

    def matmul_2d(self, x_int, w_int, chunk_rows=1024):
        """x_int (m,k) int, w_int (k,n) int -> (m,n) float64 exact MAC."""
        return np.matmul(x_int.astype(np.float64),
                         w_int.astype(np.float64))


class HttpBackend(object):
    """POST matmuls to the Gazelle compass server over HTTP."""

    name = "http"

    def __init__(self, host="127.0.0.1", port=8000, chunk_rows=1024,
                 timeout=120, reps=1):
        self.host = host
        self.port = port
        self.chunk_rows = chunk_rows
        self.timeout = timeout
        self.reps = reps
        self.calls = 0
        self.total_rows = 0

    def _b64_body(self, x_u8, weight_id=None, w_i8=None):
        """Compact base64 request body (JSON list payload is ~5x larger and
        tunnel-bandwidth-bound; b64 keeps correctness, both arrays exact)."""
        import base64
        import json
        if weight_id is not None:
            return json.dumps({
                "act_b64": base64.b64encode(
                    np.ascontiguousarray(x_u8, dtype=np.uint8).tobytes()
                ).decode("ascii"),
                "act_shape": list(x_u8.shape),
                "weight_id": weight_id,
            }).encode("utf-8")
        k = x_u8.shape[1]
        return json.dumps({
            "act_b64": base64.b64encode(
                np.zeros(k, dtype=np.uint8).tobytes()
            ).decode("ascii"),
            "act_shape": [1, k],
            "weight_b64": base64.b64encode(
                np.ascontiguousarray(w_i8, dtype=np.int8).tobytes()
            ).decode("ascii"),
            "weight_shape": list(w_i8.shape),
        }).encode("utf-8")

    def _post(self, payload):
        import json
        import urllib.request

        req = urllib.request.Request(
            "http://%s:%d/matmul" % (self.host, self.port),
            data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if "error" in data:
            raise RuntimeError("server error: %s" % data["error"])
        return data

    def matmul_2d(self, x_int, w_int):
        """x_int (m,k) int, w_int (k,n) int -> (m,n) float64 MAC (hw approx).

        The weight matrix (fixed during inference) is uploaded once and
        referenced by md5 id on subsequent calls; if reps>1 the same matmul
        is repeated and averaged (noise reduction ~1/sqrt(reps)).
        """
        import hashlib
        import json

        m, k = x_int.shape
        kn, n = w_int.shape
        assert k == kn
        w_key = hashlib.md5(w_int.astype(np.int8).tobytes()).hexdigest()
        w_uploaded = getattr(self, "_w_uploaded", set())
        if w_key not in w_uploaded:
            self._w_uploaded = w_uploaded
            w_uploaded.add(w_key)
            self._post(self._b64_body(
                np.zeros((1, k), dtype=np.uint8), w_i8=w_int.astype(np.int8)))
        out = np.zeros((m, n), dtype=np.float64)
        for start in range(0, m, self.chunk_rows):
            end = min(start + self.chunk_rows, m)
            payload = self._b64_body(
                x_int[start:end].astype(np.uint8), weight_id=w_key)
            acc = None
            for _ in range(self.reps):
                data = self._post(payload)
                arr = np.array(data["data"], dtype=np.float64).reshape(end - start, n)
                acc = arr if acc is None else acc + arr
                self.calls += 1
            out[start:end] = acc / float(self.reps)
            self.total_rows += end - start
        return out


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class GazelleOpticalEngine(OL.OpticalEngine):
    """OpticalEngine with osimulator call sites replaced by a 2-D backend.

    Optional per-channel affine correction of the raw hardware MAC:
    correction = {weight_hash: (a_j, b_j)}  with
      y_corrected[:, j] = (y_hw[:, j] - b_j) / a_j
    keyed by md5 of the int8 weight matrix (fixed during inference).
    """

    def __init__(self, backend, correction=None, verbose=False):
        # super().__init__(use_real=True) would set use_real=False when
        # osimulator is missing; we force the optical path anyway because we
        # override _matmul_real/_matmul_pre_quantized below.
        super().__init__(use_real=True, verbose=verbose)
        self.use_real = True
        self.backend = backend
        self.engine_label = backend.name
        self.correction = correction or {}

    @staticmethod
    def _weight_hash(w_int):
        import hashlib
        return hashlib.md5(w_int.astype(np.int8).tobytes()).hexdigest()

    def _correct(self, raw, w_int):
        """raw (m,n) float MAC -> corrected (m,n); no-op if no correction."""
        if not self.correction:
            return raw
        key = self._weight_hash(w_int)
        if key not in self.correction:
            return raw
        a_j, b_j = self.correction[key]
        a_j = np.asarray(a_j, dtype=np.float64)
        b_j = np.asarray(b_j, dtype=np.float64)
        a_j = np.where(np.abs(a_j) < 1e-9, 1.0, a_j)
        return (raw - b_j.reshape(1, -1)) / a_j.reshape(1, -1)

    # -- override: 2-D compass semantics (m,k)@(k,n), dequant identical to baseline --
    def _matmul_real(self, input_matrix, weight_matrix, input_bit, weight_bit):
        b, m, k = input_matrix.shape
        _, n = weight_matrix.shape

        # input: unsigned affine, per-tensor (identical to baseline)
        input_int, in_scale, in_zp = OL.quantize_to_int(
            input_matrix.reshape(-1, k), input_bit, signed=False)
        input_int = input_int.reshape(b, m, k)

        # weight: signed symmetric, per-channel (identical to baseline)
        qmax = 2 ** (weight_bit - 1) - 1
        w_abs_max = weight_matrix.abs().max(dim=0)[0].clamp(min=1e-8)
        w_scale = w_abs_max / qmax
        weight_int = (weight_matrix / w_scale.unsqueeze(0)).round() \
            .clamp(-qmax, qmax).to(torch.int32)

        t0 = time.time()
        x_np = input_int.cpu().numpy().reshape(b * m, k)   # int32
        w_np = weight_int.cpu().numpy()                    # int32 (k,n)
        raw = self.backend.matmul_2d(x_np, w_np)           # (b*m, n) MAC units
        raw = self._correct(raw, w_np)
        result_int = torch.from_numpy(raw).float().reshape(b, m, n)

        col_sum_w = weight_int.float().sum(dim=0)
        w_s = w_scale.view(1, 1, -1)
        col_s = col_sum_w.view(1, 1, -1)
        result = in_scale * w_s * result_int + in_zp * w_s * col_s
        self.stats["total_time"] += time.time() - t0
        return result

    def matmul_pre_quantized(self, input_int, weight_int, in_scale, in_zp,
                             w_scale, w_zp=None):
        squeeze_batch = False
        if input_int.dim() == 2:
            input_int = input_int.unsqueeze(0)
            squeeze_batch = True
        b, m, k = input_int.shape
        _, n = weight_int.shape

        t0 = time.time()
        x_np = input_int.cpu().numpy().reshape(b * m, k)
        w_np = weight_int.cpu().numpy()
        raw = self.backend.matmul_2d(x_np, w_np)
        raw = self._correct(raw, w_np)
        result_int = torch.from_numpy(raw).float().reshape(b, m, n)

        col_sum_w = weight_int.float().sum(dim=0)
        w_s = w_scale.view(1, 1, -1)
        col_s = col_sum_w.view(1, 1, -1)
        result = in_scale * w_s * result_int + in_zp * w_s * col_s
        if w_zp is not None and w_zp.abs().max() > 1e-6:
            row_sum_x = input_int.float().sum(dim=-1, keepdim=True)
            result = result + in_scale * w_zp.view(1, 1, -1) * w_s * row_sum_x
        if squeeze_batch:
            result = result.squeeze(0)
        self.stats["calls"] += 1
        self.stats["total_time"] += time.time() - t0
        return result


# ---------------------------------------------------------------------------
# Model (copied from src/scripts/optic_inference_int8.py — same as baseline)
# ---------------------------------------------------------------------------

class OpticSpaceNetV1_INT8(nn.Module):
    """Model 2 Phase4 v3: SpaceNet V1, bias=False, int8 QAT."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes, bias=False),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.classifier(x)


def load_state(path, model):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    own = model.state_dict()
    filtered = {k: v for k, v in sd.items()
                if k in own and tuple(own[k].shape) == tuple(v.shape)}
    model.load_state_dict(filtered, strict=False)
    print("[load_state] %d/%d tensors loaded from %s"
          % (len(filtered), len(sd), os.path.basename(path)))


class BaselineVGG(nn.Module):
    """Model 1 Phase4 v3: flat VGG, all bias=False, BN kept.

    (from src/scripts/optic_inference_int8_model1.py — baseline-identical)
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8 * 8, 256, bias=False)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes, bias=False)

    def forward(self, x):
        x = torch.relu(self.bn1_1(self.conv1_1(x)))
        x = torch.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2_1(self.conv2_1(x)))
        x = torch.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        x = torch.relu(self.bn3_1(self.conv3_1(x)))
        x = torch.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def revert_optic_to_conv2d(model, conv_name):
    """Restore an OpticConv2d to a native nn.Conv2d (variant B: conv3_2)."""
    from optic_layers import OpticConv2d
    old = getattr(model, conv_name)
    if not isinstance(old, OpticConv2d):
        return
    native = nn.Conv2d(old.in_channels, old.out_channels,
                       kernel_size=old.kernel_size, stride=old.stride,
                       padding=old.padding, dilation=old.dilation,
                       groups=old.groups, bias=(old.bias is not None))
    native.weight = nn.Parameter(old.weight.data.clone())
    if old.bias is not None:
        native.bias = nn.Parameter(old.bias.data.clone())
    setattr(model, conv_name, native)


class MiniVGG(nn.Module):
    """Model 4 MiniVGG-GAP (from src/training/model4_minivgg_gap_phase4_v3.py).

    7× Conv2d(3×3) + GAP head, ~260K params; head Linear has bias.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(48, 72, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(72), nn.ReLU(inplace=True),
            nn.Conv2d(72, 72, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(72), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(72, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(96, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)


class J1Arch(nn.Module):
    """Model 5/6 (J1 家族) — 与 eurosat_research/src/models.py MiniVGG 同构。

    Model 6 (纯 J1): stem k3 s2 → 全 1×1 (1,2,2) → GAP head(128→128→10)
    Model 5 (J1-RF+): stem k5 s2 → stage2 3×3×2 (RF 49px) → GAP head
    光计算层 = stage1/2/3 的 5 个 conv (stem/head FP32 电计算)。
    模块路径与训练端一致 (stem.0 / stage1.0 / stage2.0 / stage2.3 / stage3.0 /
    stage3.3 / head.2 / head.4), load_state 直接匹配 v8 权重。
    """

    def __init__(self, num_classes=10, channels=(16, 32, 64, 128),
                 kernels=(1, 1, 1), stem_kernel=3, head_dims=(128,)):
        super().__init__()
        C0, C1, C2, C3 = channels
        k1, k2, k3 = kernels

        def stage(cin, cout, k, depth, pool=True):
            layers = []
            for i in range(depth):
                layers += [
                    nn.Conv2d(cin if i == 0 else cout, cout,
                              kernel_size=k, padding=k // 2, bias=False),
                    nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.stem = nn.Sequential(
            nn.Conv2d(3, C0, kernel_size=stem_kernel, stride=2,
                      padding=stem_kernel // 2, bias=False),
            nn.BatchNorm2d(C0), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.stage1 = stage(C0, C1, k1, 1, pool=True)
        self.stage2 = stage(C1, C2, k2, 2, pool=True)
        self.stage3 = stage(C2, C3, k3, 2, pool=False)

        head_layers = []
        in_dim = C3
        for hd in head_dims:
            head_layers.append(nn.Linear(in_dim, hd, bias=True))
            head_layers.append(nn.ReLU(inplace=True))
            in_dim = hd
        head_layers.append(nn.Linear(in_dim, num_classes, bias=True))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  *head_layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)


# model registry: name -> (class, default weight file, extra)
MODEL_REGISTRY = {
    "model2": (OpticSpaceNetV1_INT8, "spacenet_v1_phase4_v3_int8.pth", None),
    "model3": (OpticSpaceNetV1_INT8, "spacenet_v2_phase4_v3_int8.pth", None),
    "model1a": (BaselineVGG, "baseline_vgg_phase4_v3_int8.pth", None),
    "model1b": (BaselineVGG, "baseline_vgg_phase4_v3_int8_vB.pth",
                "conv3_2"),
    "model4": (MiniVGG, "minivgg_gap_phase4_v3_int8.pth", None),
    # Model 5/6 (J1 家族, v8 漂移鲁棒权重; 权重复制于 train-test/weights/)
    "model5": (lambda nc=10: J1Arch(nc, kernels=(1, 3, 1), stem_kernel=5),
               "m5_j1rf_stem5_v8probe15.pth", None),
    "model6": (lambda nc=10: J1Arch(nc, kernels=(1, 1, 1), stem_kernel=3),
               "m6_j1_v8probe15.pth", None),
    "model7": (lambda nc=10: J1Arch(nc, channels=(12, 24, 48, 96),
                                    kernels=(1, 1, 1), stem_kernel=3),
               "m7_j1w075_v8probe15.pth", None),
    "model8": (lambda nc=10: J1Arch(nc, channels=(16, 32, 64, 128),
                                    kernels=(1, 1, 1), stem_kernel=5),
               "m8_rf_stem5_v8probe15.pth", None),
}


def build_model(weight_path, backend, correction=None,
                model_name="model2", keep_first_conv_electronic=True):
    """Baseline-compatible optical model (stem electronic, rest optical int8).

    model_name selects the architecture/weights (model2/model3/model1a/model1b).
    """
    if model_name not in MODEL_REGISTRY:
        raise SystemExit("unknown model %s (want model2/model3/model1a/model1b)"
                         % model_name)
    cls, default_weight, revert_conv = MODEL_REGISTRY[model_name]
    if weight_path is None:
        weight_path = os.path.join(_TS, "weights", default_weight)
    engine = GazelleOpticalEngine(backend, correction=correction)
    model = cls()
    load_state(weight_path, model)
    # J1 家族 (model5+): v8 训练 head_fp32/stem_fp32 -> head FC 保留电计算
    # (convert_linear=False 防止 head.2/head.4 被转 OpticLinear, 与训练语义对齐)
    head_elec = model_name.startswith("model5") or model_name.startswith("model6") \
        or model_name.startswith("model7") or model_name.startswith("model8")
    OL.build_optical_model(model, engine, pad_to_8=True,
                           input_bit=8, weight_bit=8,
                           keep_first_conv_electronic=keep_first_conv_electronic,
                           convert_linear=not head_elec)
    if revert_conv:
        revert_optic_to_conv2d(model, revert_conv)
    model.eval()
    return model, engine
