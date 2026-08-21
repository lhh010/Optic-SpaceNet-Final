#!/bin/bash
# Small-scale end-to-end validation of the whole migration flow for one model:
#   1) fresh per-layer calibration on the real board (REP=4 averaged fits)
#   2) N-image real-Gazelle inference with the fresh correction
#   3) numpy clean reference + gap report
# Usage:  MODEL=model3 N=200 REP=4 BATCH=8 bash run_small.sh
#         MODEL=model1a N=20  REP=1 BATCH=1 bash run_small.sh   (heavy model)
set -e
cd "$(dirname "$0")"
MODEL=${MODEL:-model2}
N=${N:-200}
REP=${REP:-4}
BATCH=${BATCH:-8}
CALIB_IMGS=${CALIB_IMGS:-40}
calib_file="small_calib_${MODEL}.npz"

echo "===== [1/3] model=$MODEL fresh calibration (${CALIB_IMGS} imgs, REP=4) ====="
bash run_calib.sh MODEL=$MODEL LIMIT=$CALIB_IMGS BATCH=$BATCH REP=4 \
  CALIB_OUT=$calib_file | tee "small_calib_${MODEL}.log"

echo "===== [2/3] model=$MODEL real-Gazelle inference (${N} imgs, REP=$REP) ====="
bash run_client.sh MODEL=$MODEL BACKEND=http LIMIT=$N BATCH=$BATCH REP=$REP \
  CORRECTION=$calib_file | tee "small_eval_${MODEL}.log"

echo "===== [3/3] model=$MODEL summary ====="
grep -E "FINAL http accuracy|FINAL numpy reference|HW vs clean-ref gap" \
  "small_eval_${MODEL}.log"
