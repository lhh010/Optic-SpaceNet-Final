# -*- coding: utf-8 -*-
"""Canonical Gazelle path-B board orchestration used by release gates.

The interactive per-layer visualisation intentionally uses path A (HTTP
``/matmul``), because path B only exposes batch logits.  Release checks must
not use that slower visualisation transport: final/main runs the canonical
board-side runners over SSH and downloads logits with SCP, so this module does
the same.
"""
import os
import re
import shlex
import subprocess
import tempfile
import threading
import time
import uuid

import numpy as np


class BoardUnavailable(RuntimeError):
    pass


BOARD_USER = os.environ.get("GAZELLE_SSH_USER", "uisrc")
BOARD_PASSWORD = os.environ.get("GAZELLE_SSH_PASSWORD", "5182")
BOARD_J1 = os.environ.get("GAZELLE_BOARD_J1", "/home/uisrc/j1")
BOARD_MNIST = os.environ.get("GAZELLE_BOARD_MNIST", "/home/uisrc/mnist")
PROBE_TTL_S = float(os.environ.get("GAZELLE_BOARD_PROBE_TTL", "10"))
CALIB_MAX_AGE_S = float(os.environ.get("GAZELLE_CALIB_MAX_AGE", "1200"))

BOARD_WEIGHTS = {
    "model9": os.environ.get("GAZELLE_BOARD_WEIGHT_9", "weights_w075ds3"),
    "model10": os.environ.get("GAZELLE_BOARD_WEIGHT_10", "weights_m10_5400"),
}

_probe_cache = {}
_probe_lock = threading.Lock()


def board_host():
    """Serial mode discovers an IP first; otherwise use the new direct IP."""
    return (os.environ.get("GAZELLE_SERIAL_HOST") or
            os.environ.get("GAZELLE_HOST") or "192.168.31.158")


def _ssh_base(connect_timeout=15):
    return [
        "sshpass", "-e", "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=%d" % int(connect_timeout),
        "-o", "ServerAliveInterval=15",
        "%s@%s" % (BOARD_USER, board_host()),
    ]


def _env():
    env = os.environ.copy()
    env["SSHPASS"] = BOARD_PASSWORD
    return env


def ssh_run(command, timeout=3600, connect_timeout=15):
    """Run one non-interactive command and return stdout.

    ``compass_sdk`` mutates argv, so callers pass runner parameters through
    environment variables inside the remote command, never as positionals.
    """
    cmd = _ssh_base(connect_timeout) + [command]
    try:
        proc = subprocess.run(
            cmd, env=_env(), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise BoardUnavailable("缺少 sshpass/ssh 客户端: %s" % exc)
    except subprocess.TimeoutExpired:
        raise BoardUnavailable(
            "SSH 命令超时 (%ss): %s@%s" %
            (timeout, BOARD_USER, board_host()))
    out = proc.stdout.decode("utf-8", "replace")
    err = proc.stderr.decode("utf-8", "replace")
    if proc.returncode != 0:
        detail = (err.strip() or out.strip() or "exit %d" % proc.returncode)
        raise BoardUnavailable("SSH 失败: %s" % detail[-240:])
    return out


def ssh_sudo_run(command, timeout=3600, connect_timeout=15):
    wrapped = ("printf '%s\\n' " + shlex.quote(BOARD_PASSWORD) +
               " | sudo -S -p '' sh -c " + shlex.quote(command))
    return ssh_run(wrapped, timeout=timeout,
                   connect_timeout=connect_timeout)


def probe_board():
    """Short SSH probe with TTL cache, matching final/main's offline fix."""
    host = board_host()
    now = time.monotonic()
    cached = _probe_cache.get(host)
    if cached and now - cached[0] < PROBE_TTL_S:
        return cached[1], cached[2]
    with _probe_lock:
        cached = _probe_cache.get(host)
        if cached and now - cached[0] < PROBE_TTL_S:
            return cached[1], cached[2]
        try:
            out = ssh_sudo_run(
                "printf PROBE_OK", timeout=7, connect_timeout=5)
            ok = "PROBE_OK" in out
            detail = "SSH %s" % host if ok else "SSH probe 未返回标记"
        except Exception as exc:
            ok, detail = False, str(exc)[:180]
        _probe_cache[host] = (time.monotonic(), ok, detail)
        return ok, detail


def inspect_usage():
    """Read-only evidence for the mandatory human occupancy/cool-down check."""
    command = (
        "printf '__WHO__\\n'; who 2>/dev/null; "
        "printf '__PROCS__\\n'; "
        "ps aux | grep -iE 'run_ds3|run_mnist|compass|server_gazelle' "
        "| grep -v grep | head -20; "
        "printf '__LEDGER__\\n'; tail -20 /home/uisrc/BOARD_USAGE.md 2>/dev/null"
    )
    return ssh_run(command, timeout=15, connect_timeout=5)


def conflicting_processes():
    """Return processes that must not coexist with canonical path-B gates."""
    command = (
        "ps -eo pid,user,args | "
        "grep -iE 'server_gazelle.py|run_ds3_gazelle.py|run_mnist_gazelle.py|"
        "probe_dump_ds3.py|calibrate_(any_ds3|col).py|compass_cali' | "
        "grep -v grep | head -20")
    return ssh_run(command, timeout=12, connect_timeout=5).strip()


def _numbers(output, key):
    match = re.search(
        r"%s\s*:\s*\[\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\]" %
        re.escape(key), output, re.I)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def run_evb(timeout=600):
    """Run compass_evb_test once and parse both release measurements."""
    try:
        out = ssh_sudo_run("compass_evb_test", timeout=timeout)
    except BoardUnavailable as first:
        # Some deployed images expose only the historical sample wrapper.
        fallback = ("cd /home/uisrc/sample_code/code && "
                    "python3 evb_test_sample.py")
        try:
            out = ssh_sudo_run(fallback, timeout=timeout)
        except BoardUnavailable:
            raise first
    if _numbers(out, "ebr") is None and _numbers(out, "error_std") is None:
        fallback = ("cd /home/uisrc/sample_code/code && "
                    "python3 evb_test_sample.py")
        out = ssh_sudo_run(fallback, timeout=timeout)
    return {
        "ebr": _numbers(out, "ebr"),
        "error_std": _numbers(out, "error_std"),
        "tail": "\n".join(out.strip().splitlines()[-12:]),
    }


def _calib_env_name(model_name):
    suffix = "9" if model_name == "model9" else "10"
    return os.environ.get("GAZELLE_BOARD_CALIB_" + suffix, "")


def calibration_status(model_name):
    """Resolve model-specific/latest board calibration and enforce 20 min."""
    if model_name not in BOARD_WEIGHTS:
        raise BoardUnavailable("路径 B 放行只支持 model9/model10")
    configured = _calib_env_name(model_name)
    if configured:
        remote = configured if configured.startswith("/") else (
            BOARD_J1.rstrip("/") + "/" + configured)
        source = "显式配置"
    else:
        tag = "m9" if model_name == "model9" else "m10"
        # New calibration script embeds the model tag.  Keep legacy generic
        # files as a last resort but report that their model identity is not
        # verifiable and therefore do not release the board automatically.
        command = (
            "ls -t %s/calib_scalar_*%s*.json 2>/dev/null | head -1" %
            (shlex.quote(BOARD_J1), tag))
        remote = ssh_run(command, timeout=12, connect_timeout=5).strip()
        source = "按模型自动发现"
        if not remote:
            command = (
                "ls -t %s/calib_scalar_*.json 2>/dev/null | head -1" %
                shlex.quote(BOARD_J1))
            remote = ssh_run(command, timeout=12,
                             connect_timeout=5).strip()
            source = "旧版通用文件（模型未核验）"
    if not remote:
        return {"pass": False, "calib": "", "age_s": None,
                "detail": "板上未找到校准 json；先 fresh compass_cali + 对应模型校准"}
    command = "stat -c '%Y' " + shlex.quote(remote) + "; date +%s"
    lines = ssh_run(command, timeout=12, connect_timeout=5).splitlines()
    try:
        age_s = max(0.0, float(lines[-1]) - float(lines[-2]))
    except (IndexError, ValueError):
        return {"pass": False, "calib": remote, "age_s": None,
                "detail": "无法读取校准文件时间，禁止自动放行"}
    identity_ok = source != "旧版通用文件（模型未核验）"
    fresh = age_s <= CALIB_MAX_AGE_S
    ok = identity_ok and fresh
    detail = "%s；年龄 %.1f 分钟（上限 %.0f 分钟）" % (
        source, age_s / 60.0, CALIB_MAX_AGE_S / 60.0)
    if not identity_ok:
        detail += "；请设置 GAZELLE_BOARD_CALIB_9/10"
    return {"pass": bool(ok), "calib": remote, "age_s": age_s,
            "detail": detail}


def run_mnist(limit=1000):
    command = (
        "cd %s && MNIST_METHOD=dsq MNIST_LIMIT=%d "
        "python3 run_mnist_gazelle.py" %
        (shlex.quote(BOARD_MNIST), int(limit)))
    out = ssh_sudo_run(command, timeout=900)
    acc_match = re.search(
        r"FINAL\s+.*accuracy\s*\([^)]*\):\s*([0-9.]+)%", out)
    ref_match = re.search(
        r"NumPy reference accuracy\s*\([^)]*\):\s*([0-9.]+)%", out)
    if not acc_match or not ref_match:
        raise BoardUnavailable("MNIST runner 未输出 hw/ref accuracy: %s" %
                               "\n".join(out.splitlines()[-8:]))
    acc, ref = float(acc_match.group(1)), float(ref_match.group(1))
    return {"acc": acc, "ref": ref, "gap": abs(acc - ref)}


def _scp_from(remote_path, local_path, timeout=45):
    cmd = [
        "sshpass", "-e", "scp", "-q",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=8",
        "%s@%s:%s" % (BOARD_USER, board_host(), remote_path), local_path,
    ]
    try:
        proc = subprocess.run(
            cmd, env=_env(), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise BoardUnavailable("SCP 拉取 logits 失败: %s" % exc)
    if proc.returncode:
        err = proc.stderr.decode("utf-8", "replace").strip()
        raise BoardUnavailable("SCP 拉取 logits 失败: %s" % err[-200:])


def run_ds3(model_name, offset, limit, calib_json=None):
    """Run canonical path B and return board logits downloaded through SCP."""
    if model_name not in BOARD_WEIGHTS:
        raise BoardUnavailable("路径 B 放行只支持 model9/model10")
    token = "%d_%s" % (os.getpid(), uuid.uuid4().hex[:10])
    remote_logits = "/tmp/ds3_gate_%s.npy" % token
    weights = BOARD_WEIGHTS[model_name]
    envs = {
        "DS3_WEIGHTS_DIR": weights,
        "DS3_LIMIT": str(int(limit)),
        "DS3_OFFSET": str(int(offset)),
        "DS3_BATCH": str(min(int(limit), 8)),
        "DS3_LOGITS_OUT": remote_logits,
    }
    if calib_json:
        envs["DS3_CALIB"] = calib_json
    assignments = " ".join(
        "%s=%s" % (key, shlex.quote(value))
        for key, value in envs.items())
    command = "cd %s && %s python3 run_ds3_gazelle.py" % (
        shlex.quote(BOARD_J1), assignments)
    fd, local_path = tempfile.mkstemp(prefix="gazelle_gate_", suffix=".npy")
    os.close(fd)
    try:
        out = ssh_sudo_run(command, timeout=1200)
        _scp_from(remote_logits, local_path)
        logits = np.load(local_path)
    finally:
        try:
            ssh_sudo_run("rm -f %s" % shlex.quote(remote_logits),
                         timeout=10, connect_timeout=5)
        except Exception:
            pass
        try:
            os.remove(local_path)
        except OSError:
            pass
    match = re.search(r"FINAL:\s*([0-9.]+)%", out)
    return {"acc": float(match.group(1)) if match else None,
            "logits": np.asarray(logits),
            "tail": "\n".join(out.strip().splitlines()[-10:])}
