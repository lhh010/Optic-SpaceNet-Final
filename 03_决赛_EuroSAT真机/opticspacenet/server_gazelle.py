# -*- coding: utf-8 -*-
"""Optic-SpaceNet optical matmul server for the REAL Gazelle board.

Runs on the Gazelle board as root.  stdlib-only (Python 3.6).  Exposes the
compass optical matmul as an HTTP JSON endpoint so a torch client anywhere
can offload the optical layers.

Endpoints:
  GET  /health -> {"status", "tia_gain", "calls", "uptime_s"}
  POST /matmul {"act": [[..]], "weight": [[..]]}
        -> {"data": [[..]]}   (compass_matmul result, ~integer MAC units)

IMPORTANT:
  * must run as root (sudo)
  * no positional CLI args (compass_sdk mutates sys.argv); port via env
  * single-threaded on purpose: compass_matmul uses global hardware state
  * vec uint8 (0..255), weight int8 (-128..127)
"""
import base64
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

from compass_sdk.fast_calibration.compass_lib import (
    compass_init, compass_matmul, compass_matmul_advance)
from compass_sdk.fast_calibration.utils import global_var

PORT = int(os.environ.get("OPTC_PORT", "8000"))
calls = 0
t0 = time.time()

# weight cache: weight_id -> (int8 (k,n) ndarray, nbytes)
WEIGHT_CACHE = {}
MAX_CACHE = 64


def _weight_id(wgt):
    import hashlib
    return hashlib.md5(wgt.tobytes()).hexdigest()


class Handler(BaseHTTPRequestHandler):
    server_version = "OpticMatmul/1.0"

    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {
                "status": "ok",
                "tia_gain": global_var.get_value("tia_gain_scale_factor"),
                "calls": calls,
                "cached_weights": len(WEIGHT_CACHE),
                "uptime_s": round(time.time() - t0, 2),
            })
        else:
            self._send(404, {"error": "unknown route %s" % self.path})

    def do_POST(self):
        global calls
        if self.path != "/matmul":
            self._send(404, {"error": "unknown route %s" % self.path})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            if "act_b64" in payload:
                act = np.frombuffer(base64.b64decode(payload["act_b64"]),
                                    dtype=np.uint8)
                act = act.reshape(payload["act_shape"])
            else:
                act = np.array(payload["act"], dtype=np.uint8)
            if act.ndim != 2:
                raise ValueError("act must be 2-D")

            wid = payload.get("weight_id")
            if wid:
                if wid not in WEIGHT_CACHE:
                    raise ValueError("unknown weight_id %s" % wid)
                wgt = WEIGHT_CACHE[wid]
            else:
                if "weight_b64" in payload:
                    wgt = np.frombuffer(
                        base64.b64decode(payload["weight_b64"]),
                        dtype=np.int8).reshape(payload["weight_shape"])
                else:
                    wgt = np.array(payload["weight"], dtype=np.int8)
                if wgt.ndim != 2:
                    raise ValueError("weight must be 2-D")
                wid = _weight_id(wgt)
                if wid not in WEIGHT_CACHE:
                    if len(WEIGHT_CACHE) >= MAX_CACHE:
                        WEIGHT_CACHE.clear()
                    WEIGHT_CACHE[wid] = wgt
            if act.shape[1] != wgt.shape[0]:
                raise ValueError("dim mismatch %s vs %s"
                                 % (act.shape, wgt.shape))
            func = payload.get("func", "compass_matmul")
            if func == "compass_matmul":
                res = compass_matmul(act, wgt)
            elif func == "compass_matmul_advance":
                res = compass_matmul_advance(act, wgt)
            else:
                raise ValueError("unknown func %s" % func)
            calls += 1
            self._send(200, {"data": np.asarray(res).tolist(),
                             "shape": list(np.asarray(res).shape),
                             "weight_id": wid})
        except Exception as e:
            self._send(500, {"error": "%s: %s" % (type(e).__name__, e)})

    def log_message(self, fmt, *args):
        pass


def main():
    print("[optic_matmul] compass_init(150) ...", flush=True)
    compass_init(150)
    print("[optic_matmul] tia_gain =", global_var.get_value(
        "tia_gain_scale_factor"), flush=True)
    srv = HTTPServer(("0.0.0.0", PORT), Handler)
    print("[optic_matmul] listening on 0.0.0.0:%d" % PORT, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
