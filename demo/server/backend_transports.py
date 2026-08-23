"""Selectable matrix transports for the interactive demo.

Gazelle's serial port is a Linux console, not a matrix-RPC protocol.  Serial
mode therefore discovers the board IP through the console, then uses the same
board /matmul HTTP service.  SSH mode carries /matmul traffic through a real
local-forward tunnel.
"""
import glob
import os
import re
import socket
import subprocess
import threading
import time
from urllib.parse import urlparse

from demo.server.gazelle_engine import HttpBackend


class TransportUnavailable(RuntimeError):
    pass


def _port_open(host, port, timeout=0.25):
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


class OsimulatorBackend(HttpBackend):
    """Gazelle-compatible /matmul endpoint served by local osimulator."""

    name = "gazelle-osim"

    def __init__(self):
        raw = os.environ.get("OPTIC_REMOTE_URL", "http://127.0.0.1:8765")
        parsed = urlparse(raw if "://" in raw else "http://" + raw)
        if not parsed.hostname:
            raise TransportUnavailable("invalid OPTIC_REMOTE_URL: %s" % raw)
        super().__init__(
            host=parsed.hostname, port=parsed.port or 8765,
            timeout=float(os.environ.get("OPTIC_TIMEOUT", "300")))
        self.name = "gazelle-osim"


class SshTunnelBackend(HttpBackend):
    """Gazelle /matmul over ssh -L to uisrc@192.168.31.158."""

    name = "gazelle-hardware-ssh"
    _lock = threading.Lock()
    _process = None
    _last_error_at = 0.0
    _last_error = ""

    def __init__(self):
        self.board_host = os.environ.get("GAZELLE_HOST", "192.168.31.158")
        self.board_port = int(os.environ.get("GAZELLE_PORT", "8000"))
        self.user = os.environ.get("GAZELLE_SSH_USER", "uisrc")
        self.password = os.environ.get("GAZELLE_SSH_PASSWORD", "5182")
        self.local_port = int(os.environ.get("GAZELLE_SSH_LOCAL_PORT", "18080"))
        super().__init__(
            host="127.0.0.1", port=self.local_port,
            timeout=float(os.environ.get("GAZELLE_TIMEOUT", "300")))
        self.name = "gazelle-hardware-ssh"

    def _ensure_tunnel(self):
        if _port_open("127.0.0.1", self.local_port):
            return
        with self._lock:
            if _port_open("127.0.0.1", self.local_port):
                return
            if (time.monotonic() - self.__class__._last_error_at < 5.0 and
                    self.__class__._last_error):
                raise TransportUnavailable(self.__class__._last_error)
            proc = self.__class__._process
            if proc is not None and proc.poll() is None:
                return
            env = os.environ.copy()
            env["SSHPASS"] = self.password
            cmd = [
                "sshpass", "-e", "ssh", "-N",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "ExitOnForwardFailure=yes",
                "-o", "ConnectTimeout=8",
                "-o", "ServerAliveInterval=15",
                "-L", "127.0.0.1:%d:127.0.0.1:%d" %
                (self.local_port, self.board_port),
                "%s@%s" % (self.user, self.board_host),
            ]
            try:
                proc = subprocess.Popen(
                    cmd, env=env, stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except (OSError, ValueError) as exc:
                raise TransportUnavailable("cannot start SSH tunnel: %s" % exc)
            self.__class__._process = proc
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                if _port_open("127.0.0.1", self.local_port):
                    self.__class__._last_error = ""
                    return
                if proc.poll() is not None:
                    break
                time.sleep(0.15)
            if proc.poll() is None:
                proc.terminate()
            message = "SSH unavailable: %s@%s (local tunnel :%d)" % (
                self.user, self.board_host, self.local_port)
            self.__class__._last_error_at = time.monotonic()
            self.__class__._last_error = message
            raise TransportUnavailable(message)

    def _post(self, payload):
        self._ensure_tunnel()
        return super()._post(payload)


class SerialBootstrapBackend(HttpBackend):
    """Discover Gazelle through its 115200 8N1 console, then call /matmul."""

    name = "gazelle-hardware-serial"
    _lock = threading.Lock()

    def __init__(self):
        self.device = os.environ.get("GAZELLE_SERIAL_DEVICE", "")
        self.console_user = os.environ.get("GAZELLE_SERIAL_USER", "uisrc")
        self.console_password = os.environ.get("GAZELLE_SERIAL_PASSWORD", "root")
        self.board_port = int(os.environ.get("GAZELLE_PORT", "8000"))
        fallback_host = os.environ.get("GAZELLE_SERIAL_HOST", "")
        super().__init__(
            host=fallback_host or "127.0.0.1", port=self.board_port,
            timeout=float(os.environ.get("GAZELLE_TIMEOUT", "300")))
        self.name = "gazelle-hardware-serial"
        # A desktop launcher may already have authenticated through the host
        # serial console and discovered the IP before starting this container.
        self._resolved = bool(fallback_host)

    @staticmethod
    def _read_for(ser, seconds):
        end = time.monotonic() + seconds
        chunks = []
        while time.monotonic() < end:
            data = ser.read(4096)
            if data:
                chunks.append(data)
            else:
                time.sleep(0.05)
        return b"".join(chunks).decode("utf-8", errors="replace")

    def _discover(self):
        if self._resolved:
            return
        with self._lock:
            if self._resolved:
                return
            devices = [self.device] if self.device else (
                sorted(glob.glob("/dev/ttyUSB*")) +
                sorted(glob.glob("/dev/ttyACM*")))
            devices = [item for item in devices if item]
            if not devices:
                raise TransportUnavailable(
                    "未发现 Gazelle 串口；接入 CP210x 后重建容器以挂载 /dev/ttyUSB*")
            try:
                import serial
                with serial.Serial(
                        devices[0], 115200, timeout=0.15,
                        write_timeout=2, exclusive=True) as ser:
                    ser.write(b"\r\n")
                    text = self._read_for(ser, 0.8)
                    if re.search(r"login:\s*$", text, re.I):
                        ser.write((self.console_user + "\r\n").encode())
                        text += self._read_for(ser, 0.8)
                    if re.search(r"password:\s*$", text, re.I):
                        ser.write((self.console_password + "\r\n").encode())
                        text += self._read_for(ser, 1.0)
                    marker = "__GAZELLE_ADDR__"
                    command = (
                        "echo %s; ip -4 -o addr show scope global; "
                        "echo __GAZELLE_END__\r\n" % marker)
                    ser.write(command.encode())
                    text += self._read_for(ser, 3.0)
            except Exception as exc:
                raise TransportUnavailable(
                    "Gazelle 串口 %s 打开或登录失败: %s" %
                    (devices[0], str(exc)[:120]))
            candidates = re.findall(
                r"\binet\s+((?:\d{1,3}\.){3}\d{1,3})/", text)
            candidates = [ip for ip in candidates if not ip.startswith("127.")]
            if candidates:
                self.host = candidates[0]
            elif not os.environ.get("GAZELLE_SERIAL_HOST"):
                raise TransportUnavailable(
                    "串口已连接但未读到板卡 IPv4；请确认 console 已登录并执行 ifconfig")
            self._resolved = True

    def _post(self, payload):
        self._discover()
        return super()._post(payload)


def matrix_backend(mode):
    if mode == "osimulator":
        return OsimulatorBackend()
    if mode == "gazelle_ssh":
        return SshTunnelBackend()
    if mode == "gazelle_serial":
        return SerialBootstrapBackend()
    raise TransportUnavailable("unsupported matrix backend: %s" % mode)
