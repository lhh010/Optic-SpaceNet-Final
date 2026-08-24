# -*- coding: utf-8 -*-
"""板边(路径B)编排: 通过 SSH 在板上运行 runner, 解析结果。
demo 不再走 pathA HTTP matmul (实测 ~10s/次, 单图 90s+), 改为 pathB 板端 runner
(~3.2s/张, M10 95.33% canonical 链)。"""
import os
import re
import time

import numpy as np
import paramiko

BOARD_HOST = os.environ.get("BOARD_HOST", "192.168.31.158")
BOARD_USER = os.environ.get("BOARD_USER", "uisrc")
BOARD_PASS = os.environ.get("BOARD_PASS", "5182")
J1_DIR = os.environ.get("BOARD_J1", "/home/uisrc/j1")
MNIST_DIR = os.environ.get("BOARD_MNIST", "/home/uisrc/mnist")

_conn = None


def _connect(timeout=15):
    global _conn
    if _conn is None or _conn.get_transport() is None or not _conn.get_transport().is_active():
        _conn = paramiko.SSHClient()
        _conn.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        _conn.connect(BOARD_HOST, port=22, username=BOARD_USER,
                     password=BOARD_PASS, timeout=timeout,
                     look_for_keys=False, allow_agent=False)
    return _conn


def _quote(s):
    """单引号 quote (适合作 sh 里一个 word)。"""
    return "'" + s.replace("'", "'\\''") + "'"


def ssh_run(cmd, timeout=3600, connect_timeout=15):
    c = _connect(connect_timeout)
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    rc = stdout.channel.recv_exit_status()
    return out, err, rc


def ssh_sudo_run(cmd, timeout=3600, connect_timeout=15):
    """sudo 执行(root, 经 stdin 喂密码); 板端 runner 需 root 写 api.log。"""
    wrapped = "echo " + _quote(BOARD_PASS) + " | sudo -S sh -c " + _quote(cmd)
    return ssh_run(wrapped, timeout=timeout, connect_timeout=connect_timeout)


def run_ds3(offset, limit, calib_json=None, weights="weights_m10_5400"):
    """板上跑 ds3 (M10) 批量 [offset, offset+limit)。返回 dict。"""
    env = {"DS3_WEIGHTS_DIR": weights, "DS3_LIMIT": str(limit),
           "DS3_OFFSET": str(offset), "DS3_BATCH": str(min(limit, 8))}
    if calib_json:
        env["DS3_CALIB"] = calib_json
    envs = " ".join(k + "=" + _quote(v) for k, v in env.items())
    logits_path = "/tmp/ds3_logits_%d.npy" % int(time.time())
    envs = envs + " DS3_LOGITS_OUT=" + _quote(logits_path)
    cmd = "cd " + J1_DIR + " && " + envs + " python3 run_ds3_gazelle.py"
    t0 = time.time()
    out, err, rc = ssh_sudo_run(cmd, timeout=1200)
    elapsed = time.time() - t0
    m = re.search(r"FINAL:\s*([0-9.]+)%", out)
    acc = float(m.group(1)) if m else None
    mtrace = re.findall(r"\[\s*(\d+)/\d+\] acc=([0-9.]+)%", out)
    logits = _read_npy(remote_path=logits_path)  # (n,10) 真机 logits
    return {"acc": acc, "elapsed_s": round(elapsed, 1),
            "sec_per_img": round(elapsed / max(1, limit), 2) if acc is not None else None,
            "trace": [{"n": int(n), "acc": float(a)} for n, a in mtrace],
            "logits": logits, "rc": rc, "stderr": (err[-300:] if rc else "")}


def run_mnist(limit, method="dsq", official=False):
    """板上跑 MNIST。official=True 用官方200 (需已上传并调用 run_mnist_official.py)。"""
    if official:
        cmd = ("cd " + MNIST_DIR + " && MNIST_METHOD=" + _quote(method) +
               " MNIST_LIMIT=" + str(limit) + " python3 run_mnist_official.py")
    else:
        cmd = ("cd " + MNIST_DIR + " && MNIST_METHOD=" + _quote(method) +
               " MNIST_LIMIT=" + str(limit) + " python3 run_mnist_gazelle.py")
    t0 = time.time()
    out, err, rc = ssh_sudo_run(cmd, timeout=900)
    elapsed = time.time() - t0
    m = re.search(r"FINAL\s+.*accuracy\s*\([^)]*\):\s*([0-9.]+)%", out)
    acc = float(m.group(1)) if m else None
    nref = re.search(r"NumPy reference accuracy\s*\([^)]*\):\s*([0-9.]+)%", out)
    ref = float(nref.group(1)) if nref else None
    gap = round(abs(acc - ref), 2) if acc is not None and ref is not None else None
    return {"acc": acc, "ref": ref, "gap": gap,
            "elapsed_s": round(elapsed, 1), "rc": rc, "stderr": err[-200:]}


def _read_npy(remote_path, timeout=30):
    """从板子 SFTP 拉一个 npy 到内存并返回 np 数组。失败返回 None。"""
    import tempfile
    try:
        c = _connect()
        sftp = c.open_sftp()
        local = os.path.join(tempfile.gettempdir(), "board_" + os.path.basename(remote_path))
        sftp.get(remote_path, local)
        sftp.close()
        a = np.load(local)
        try:
            os.remove(local)
        except Exception:
            pass
        return a
    except Exception as e:
        return None


def run_calibrate(weights="weights_m10_5400", calib_out="calib_scalar_auto.json"):
    """顺序校准: compass_cali(带bringup)后 calibrate_any_ds3(标量)。必须顺序, 否则 device busy。
    返回 {ok, step, calib, err}。"""
    out1, err1, rc1 = ssh_sudo_run("compass_cali --mode-local", timeout=1800)
    if rc1 != 0:
        return {"ok": False, "step": "compass_cali", "calib": "", "err": err1[-200:]}
    envs = "DS3_WEIGHTS_DIR=" + _quote(weights) + " DS3_CALIB_OUT=" + _quote(calib_out)
    cmd = "cd " + J1_DIR + " && " + envs + " python3 calibrate_any_ds3.py"
    out2, err2, rc2 = ssh_sudo_run(cmd, timeout=900)
    ok2 = (rc2 == 0 and "saved" in out2)
    return {"ok": ok2, "step": "scalar_calib", "calib": calib_out,
            "err": ("" if ok2 else (err2[-200:] or out2[-200:]))}


def latest_calib():
    """板上 ~/j1 里最新的标量校准 json 名 (ls -t 按 mtime 新→旧)。无则 None。"""
    out, err, rc = ssh_sudo_run("ls -t " + J1_DIR + "/calib_scalar_*.json 2>/dev/null | head -1",
                                timeout=20, connect_timeout=5)
    name = out.strip()
    if name and name.endswith(".json"):
        return os.path.basename(name)
    return None


def run_ebr(timeout=600):
    """板上 compass_evb_test, 解析 EBR (两通道)。返回 (ebr1, ebr2) 或 (None,None)。"""
    out, err, rc = ssh_sudo_run("compass_evb_test", timeout=timeout)
    m = re.search(r"ebr:\s*\[\s*([0-9.]+)\s+([0-9.]+)\s*\]", out)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


_probe_cache = {"t": 0.0, "ok": False, "detail": ""}
PROBE_TTL_S = float(os.environ.get("BOARD_PROBE_TTL", "10"))
PROBE_CONNECT_S = float(os.environ.get("BOARD_PROBE_CONNECT", "5"))
PROBE_EXEC_S = float(os.environ.get("BOARD_PROBE_EXEC", "8"))


def probe_board():
    """可达性: 快速探测(独立短超时) + TTL 缓存 + 首次失败重试一次。
    板子离线/闪断时 health 轮询不阻塞堆积; 对瞬时网络抖动(慢连接/刚上电/ARP)
    更宽容, 避免"在线但抖动"被误判为不可达。"""
    now = time.time()
    if now - _probe_cache["t"] < PROBE_TTL_S:
        return _probe_cache["ok"], _probe_cache["detail"]
    ok, detail = False, ""
    for attempt in (1, 2):
        try:
            out, err, rc = ssh_sudo_run("echo PROBE_OK",
                                        timeout=PROBE_EXEC_S,
                                        connect_timeout=PROBE_CONNECT_S)
            if rc == 0 and "PROBE_OK" in out:
                ok, detail = True, "SSH " + BOARD_HOST
            else:
                detail = "连接成功但命令rc=%d" % rc
            break                       # 有连接结果就不必重试
        except Exception as e:
            detail = str(e)[:60]        # 连接层失败: 可能超时/闪断
            if attempt == 1:
                time.sleep(0.5)         # 留给刚上电/网络恢复一点时间再试一次
    if not ok and "timed out" in detail.lower():
        detail = "网络超时(板子未接网或闪断, 等待重连)"
    elif not ok and detail.startswith("连接成功"):
        pass                            # 保留 rc 信息
    elif not ok:
        detail = "SSH不可达: " + (detail or "未知错误")[:60]
    _probe_cache.update(t=now, ok=ok, detail=detail)
    return ok, detail


def upload(local_path, remote_path):
    c = _connect()
    sftp = c.open_sftp()
    sftp.put(local_path, remote_path)
    sftp.close()