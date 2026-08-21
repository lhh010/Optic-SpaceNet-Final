#!/bin/bash
# Full 5400-image real-Gazelle run with segmented re-calibration.
#
# The board's per-channel gain/offset (a_j, b_j) drift over ~1h, so we
# re-calibrate at the start of every segment and evaluate that segment with
# the fresh correction.  REP=4 averaging + fresh correction measured at
# 86.0-88.5% vs 89.0% clean reference on 200-image windows (gap 0.5-3 pts,
# varies with hardware drift timing).
#
# Usage:  SEG=600 REP=4 BATCH=8 bash run_full.sh
set -e
cd "$(dirname "$0")"
SEG=${SEG:-600}
REP=${REP:-4}
BATCH=${BATCH:-8}
CALIB_IMGS=${CALIB_IMGS:-40}
RESULT=full_results.txt
: > "$RESULT"
echo "seg  offset  limit  hw_acc  ref_acc  gap" | tee -a "$RESULT"

for ((off=0; off<5400; off+=SEG)); do
  len=$(( SEG < 5400 - off ? SEG : 5400 - off ))
  calib_file="seg_calib_${off}.npz"
  echo "===== segment offset=$off limit=$len : calibrating ..." | tee -a "$RESULT"
  bash run_calib.sh LIMIT=$CALIB_IMGS BATCH=$BATCH CALIB_OUT=$calib_file \
    > "seg_calib_${off}.log" 2>&1 || { echo "calib failed at $off"; exit 1; }
  echo "===== segment offset=$off : evaluating $len images (REP=$REP) ..." | tee -a "$RESULT"
  bash run_client.sh BACKEND=http LIMIT=$len BATCH=$BATCH REP=$REP OFFSET=$off \
    CORRECTION=$calib_file > "seg_eval_${off}.log" 2>&1 || { echo "eval failed at $off"; exit 1; }
  hw=$(grep "FINAL http accuracy" "seg_eval_${off}.log" | sed -E 's/.*: ([0-9.]+)%.*/\1/')
  ref=$(grep "FINAL numpy reference" "seg_eval_${off}.log" | sed -E 's/.*: ([0-9.]+)%.*/\1/')
  gap=$(grep "HW vs clean-ref gap" "seg_eval_${off}.log" | sed -E 's/.*: ([+-]?[0-9.]+) points.*/\1/')
  echo "seg$((off/SEG+1))  $off  $len  hw=$hw  ref=$ref  gap=$gap" | tee -a "$RESULT"
done
echo "ALL SEGMENTS DONE" | tee -a "$RESULT"
