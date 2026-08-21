#!/bin/bash
# Helper: run the per-layer calibration on the real board.
#   ./run_calib.sh LIMIT=100 CALIB_OUT=calib.npz
set -e
cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE
export WSLENV=KMP_DUPLICATE_LIB_OK:LIMIT:BATCH:WEIGHT:DATA:OPTC_HOST:OPTC_PORT:CALIB_OUT:REP:MODEL
for kv in "$@"; do export "$kv"; done
/mnt/e/anaconda3/python.exe analyze_layers.py
