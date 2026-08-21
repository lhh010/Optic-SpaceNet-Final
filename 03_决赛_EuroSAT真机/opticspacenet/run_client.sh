#!/bin/bash
# Helper: run the Optic-SpaceNet Gazelle client with proper env forwarding
# to the Windows anaconda python.  Usage:
#   ./run_client.sh BACKEND=http LIMIT=100 BATCH=1 [CORRECTION=calib.npz]
set -e
cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE
export WSLENV=KMP_DUPLICATE_LIB_OK:BACKEND:LIMIT:BATCH:WEIGHT:DATA:OPTC_HOST:OPTC_PORT:REF:CORRECTION:REP:OFFSET:MODEL:ERR_OUT
# apply explicit assignments from CLI (e.g. BACKEND=http LIMIT=100)
for kv in "$@"; do export "$kv"; done
/mnt/e/anaconda3/python.exe run_eval.py
