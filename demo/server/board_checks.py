"""Gazelle release gates aligned with Optic-SpaceNet-Final ``main``.

The four gates run through the canonical board-side path B.  Offline numpy or
the per-layer path-A visualisation transport can never release the board.
"""
import os

import numpy as np

from demo.server import board_runner, ds3net
from demo.server.gazelle_engine import NumpyBackend
from demo.server.gazelle_client import MODEL_DEFS, _get_ds3


ERROR_STD_BASELINE = np.array([
    float(os.environ.get("HW_ERROR_STD_BASE_1", "4.694")),
    float(os.environ.get("HW_ERROR_STD_BASE_2", "4.473")),
])
ERROR_STD_TOL_PCT = float(os.environ.get("HW_ERROR_STD_TOL", "2.0"))
CANARY_N = int(os.environ.get("HW_CANARY_N", "1000"))
MINIRUN_DEFAULT_N = int(os.environ.get("HW_CHECK_N", "100"))
MINIRUN_GAP_PT = float(os.environ.get("HW_MINIRUN_GAP", "2.0"))


def _failed(name, detail, value=""):
    return {"name": name, "pass": False, "value": value,
            "detail": detail}


def _blocked(name, reason):
    out = _failed(name, "未执行：" + reason)
    out["blocked"] = True
    return out


def check_ebr_values(values):
    if not values or len(values) != 2:
        return _failed("① EBR ≥ 8", "compass_evb_test 未输出两通道 EBR")
    ebr = np.asarray(values, dtype=np.float64)
    ok = bool(np.all(ebr >= 8.0))
    return {"name": "① EBR ≥ 8", "pass": ok,
            "value": "%.3f / %.3f" % tuple(ebr),
            "detail": "板端 compass_evb_test 自动测量；两个通道都必须 ≥8"}


def check_error_std_values(values):
    if not values or len(values) != 2:
        return _failed(
            "② error_std 对健康基线偏差 < +%.1f%%" % ERROR_STD_TOL_PCT,
            "compass_evb_test 未输出两通道 error_std")
    measured = np.asarray(values, dtype=np.float64)
    deviation = (measured - ERROR_STD_BASELINE) / ERROR_STD_BASELINE * 100.0
    # The project record explicitly treats a lower noise value as an
    # improvement.  Only degradation beyond the tolerance blocks release.
    ok = bool(np.all(deviation < ERROR_STD_TOL_PCT))
    return {
        "name": "② error_std 对健康基线偏差 < +%.1f%%" % ERROR_STD_TOL_PCT,
        "pass": ok,
        "value": "%.3f / %.3f（%+.2f%% / %+.2f%%）" %
                 (measured[0], measured[1], deviation[0], deviation[1]),
        "detail": "健康基线 %.3f / %.3f；低于基线视为改善，任一通道恶化超阈值即失败" %
                  tuple(ERROR_STD_BASELINE),
    }


def check_evb():
    """Run one EVB workload and produce gates ① and ② from the same sample."""
    try:
        result = board_runner.run_evb()
    except Exception as exc:
        reason = str(exc)[:220]
        return (_failed("① EBR ≥ 8", reason),
                _failed("② error_std 对健康基线", reason))
    return (check_ebr_values(result.get("ebr")),
            check_error_std_values(result.get("error_std")))


def check_canary():
    """final/main: board-side DSQ MNIST n=1000, hw/ref gap <0.5pt."""
    try:
        result = board_runner.run_mnist(CANARY_N)
    except Exception as exc:
        return _failed("③ MNIST canary gap < 0.5pt", str(exc)[:220])
    gap = float(result["gap"])
    return {
        "name": "③ MNIST canary gap < 0.5pt",
        "pass": bool(gap < 0.5),
        "value": "hw %.2f%% vs ref %.2f%%，gap %.2fpt（n=%d）" %
                 (result["acc"], result["ref"], gap, CANARY_N),
        "detail": "板端 run_mnist_gazelle.py 的 DSQ 三层 MLP 与同量化 NumPy 参考",
    }


def check_minirun(model_name="model10", n=MINIRUN_DEFAULT_N,
                  images=None, labels=None, calib_json=None):
    """final/main: path-B board accuracy vs local quantized reference <2pt."""
    if model_name not in ("model9", "model10"):
        return _failed("④ EuroSAT mini-run 偏差 < 2pt",
                       "路径 B canonical runner 只支持 M9/M10")
    if images is None or labels is None:
        return _failed("④ EuroSAT mini-run 偏差 < 2pt",
                       "缺少 test200 数据；先运行 tools/make_test200.py")
    n = max(1, min(int(n), 500, len(images), len(labels)))
    try:
        ws, meta, _ = _get_ds3(model_name)
        batch = np.asarray(images[:n], dtype=np.float64)
        targets = np.asarray(labels[:n], dtype=np.int64)
        ref_logits = ds3net.forward(
            batch, ws, meta, NumpyBackend(), calib_col=None,
            head_elec=MODEL_DEFS[model_name]["head_elec"])
        result = board_runner.run_ds3(
            model_name, 0, n, calib_json=calib_json)
        hw_logits = np.asarray(result["logits"])
        if hw_logits.ndim != 2 or hw_logits.shape[0] < n:
            raise ValueError("板端 logits 形状异常: %s" % (hw_logits.shape,))
        hw_acc = float(np.mean(np.argmax(hw_logits[:n], 1) == targets)) * 100.0
        ref_acc = float(np.mean(np.argmax(ref_logits[:n], 1) == targets)) * 100.0
    except Exception as exc:
        return _failed("④ EuroSAT mini-run 偏差 < 2pt", str(exc)[:220])
    gap = abs(hw_acc - ref_acc)
    return {
        "name": "④ EuroSAT mini-run 偏差 < %.1fpt" % MINIRUN_GAP_PT,
        "pass": bool(gap < MINIRUN_GAP_PT),
        "value": "hw %.1f%% vs ref %.1f%%，gap %.1fpt（n=%d）" %
                 (hw_acc, ref_acc, gap, n),
        "detail": "板端 run_ds3_gazelle.py 路径 B + SCP logits，对比本地同量化 NumPy 参考",
    }


def preflight(model_name="model10", confirmed_idle=False):
    """Non-negotiable SOP prerequisites that precede the four workloads."""
    occupancy = {
        "name": "前置：无人占用且已冷却 ≥5min",
        "pass": bool(confirmed_idle),
        "value": "已人工确认" if confirmed_idle else "未确认",
        "detail": "先查看 who / ps 与 BOARD_USAGE.md；本服务不能可靠判断进程归属",
    }
    if not confirmed_idle:
        conflicts = {
            "name": "前置：path-B 无冲突进程", "pass": False,
            "value": "未检查", "detail": "未确认板卡空闲前不连接板卡",
        }
        calibration = {
            "name": "前置：对应模型校准 ≤20min",
            "pass": False, "value": "未检查",
            "detail": "未确认板卡空闲前不会发起任何板端工作负载",
        }
        return [occupancy, conflicts, calibration], None
    ok, detail = board_runner.probe_board()
    if not ok:
        conflicts = {
            "name": "前置：path-B 无冲突进程", "pass": False,
            "value": "板卡不可达", "detail": detail,
        }
        calibration = {
            "name": "前置：对应模型校准 ≤20min", "pass": False,
            "value": "板卡不可达", "detail": detail,
        }
        return [occupancy, conflicts, calibration], None
    try:
        process_text = board_runner.conflicting_processes()
        conflicts = {
            "name": "前置：path-B 无冲突进程",
            "pass": not bool(process_text),
            "value": "无" if not process_text else "检测到占用",
            "detail": ("未发现 server_gazelle/runner/校准进程" if not process_text
                       else process_text[:500]),
        }
    except Exception as exc:
        conflicts = {
            "name": "前置：path-B 无冲突进程", "pass": False,
            "value": "检查失败", "detail": str(exc)[:220],
        }
    try:
        status = board_runner.calibration_status(model_name)
        calibration = {
            "name": "前置：对应模型校准 ≤20min",
            "pass": bool(status["pass"]),
            "value": os.path.basename(status.get("calib") or "未找到"),
            "detail": status["detail"],
        }
        calib = status.get("calib") if status["pass"] else None
    except Exception as exc:
        calibration = {
            "name": "前置：对应模型校准 ≤20min", "pass": False,
            "value": "检查失败", "detail": str(exc)[:220],
        }
        calib = None
    return [occupancy, conflicts, calibration], calib


def all_checks(model_name="model10", images=None, labels=None,
               n=MINIRUN_DEFAULT_N, confirmed_idle=False):
    prereqs, calib_json = preflight(model_name, confirmed_idle)
    if not all(item["pass"] for item in prereqs):
        reason = next(item["detail"] for item in prereqs if not item["pass"])
        checks = [
            _blocked("① EBR ≥ 8", reason),
            _blocked("② error_std 对健康基线", reason),
            _blocked("③ MNIST canary gap < 0.5pt", reason),
            _blocked("④ EuroSAT mini-run 偏差 < 2pt", reason),
        ]
    else:
        ebr, error_std = check_evb()
        # Do not burn another 5-10 minutes when the physical window is already
        # known bad.  This also prevents a failed EBR check from heating it.
        if not (ebr["pass"] and error_std["pass"]):
            reason = "EBR/error_std 未通过"
            checks = [ebr, error_std,
                      _blocked("③ MNIST canary gap < 0.5pt", reason),
                      _blocked("④ EuroSAT mini-run 偏差 < 2pt", reason)]
        else:
            canary = check_canary()
            mini = (check_minirun(
                model_name, n=n, images=images, labels=labels,
                calib_json=calib_json) if canary["pass"] else
                _blocked("④ EuroSAT mini-run 偏差 < 2pt",
                         "MNIST canary 未通过"))
            checks = [ebr, error_std, canary, mini]
    all_pass = (all(item["pass"] for item in prereqs) and
                all(item["pass"] for item in checks))
    return {
        "all_pass": all_pass,
        "preflight": prereqs,
        "checks": checks,
        "path": "gazelle-hardware:pathB",
        "sop_hint": [
            "旧跳板机路径已失效；当前仅直连 uisrc@192.168.31.158",
            "fresh compass_cali + 对应模型校准后，四项判据背靠背执行",
            "单次运行 15-20min；空闲冷却 ≥5min；校准超过20min必须重做",
            "路径 A 仅用于任意图片逐层可视化，不作为正式放行结果",
        ],
    }
