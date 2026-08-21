#!/bin/bash
# ==============================================================================
# CICC1003564 | M1 (Baseline VGG 变体 A) 部分验证受控驱动 (2026-08-17)
# 调度策略: 30min 预算(校准扣除)/6min 冷却/每3段 chip cali/0% 自愈/降准 ALERT
# ==============================================================================
cd /home/uisrc/j1
LOG=/home/uisrc/m1_ccic.log
STATE=/home/uisrc/j1/m1_state.txt
PW=5182
EXPECT=${M1_EXPECT:-95.0}
WD=/home/uisrc/j1/weights_m1_5400
SCAL=/home/uisrc/j1/calib_scalar_m1ccic.json
COL=/home/uisrc/j1/calib_col_m1ccic.json
PREF=probe_m1ccic_

log(){ echo "[$(date '+%F %T')] $*" >> $LOG; }
state(){ echo "$*" > $STATE; }

run_seg(){
  local OFF=$1 N=$2 LIMIT=$3
  T0=$(date +%s)
  log "=== [M1 seg#$N] off=$OFF probe+calib start ==="
  echo $PW | sudo -S -p X env PYTHONIOENCODING=utf-8 M1_WEIGHTS_DIR=$WD PROBE_OUT_PREFIX=$PREF \
    python3 probe_dump_m1.py >> $LOG 2>&1
  echo $PW | sudo -S -p X env PYTHONIOENCODING=utf-8 M1_WEIGHTS_DIR=$WD M1_CALIB_OUT=$SCAL \
    python3 calibrate_any_m1.py >> $LOG 2>&1
  echo $PW | sudo -S -p X env PYTHONIOENCODING=utf-8 CALIB_COL_PAIRS_DIR=/home/uisrc/j1 \
    CALIB_COL_OUT=$COL CALIB_COL_PREFIX=$PREF \
    CALIB_COL_LAYERS=conv1_2,conv2_1,conv2_2,conv3_1,conv3_2,fc1,fc2 CALIB_COL_SCALAR=$SCAL \
    python3 calibrate_col.py >> $LOG 2>&1
  T1=$(date +%s); CMIN=$(( (T1-T0)/60 )); [ $CMIN -lt 1 ] && CMIN=1
  log "=== [M1 seg#$N] off=$OFF calib ${CMIN}min -> run (limit=$LIMIT) ==="
  echo $PW | sudo -S -p X env PYTHONIOENCODING=utf-8 M1_WEIGHTS_DIR=$WD M1_OFFSET=$OFF M1_LIMIT=$LIMIT M1_BATCH=1 \
    M1_CALIB=$SCAL M1_CALIB_COL=$COL M1_LOGITS_OUT=/home/uisrc/j1/logits_${PREF}_off$OFF.npy \
    python3 run_m1_gazelle.py >> $LOG 2>&1
  T2=$(date +%s)
  ACC=$(grep -a 'FINAL:' $LOG | tail -1 | grep -aoE '[0-9]+[.][0-9]+' | tail -1)
  [ -z "$ACC" ] && ACC=-1
  if awk "BEGIN{exit !($ACC == 0.00)}"; then
    log "!!! [M1 seg#$N] off=$OFF 0% -> self-heal re-probe+calib ==="
    echo $PW | sudo -S -p X env PYTHONIOENCODING=utf-8 M1_WEIGHTS_DIR=$WD PROBE_OUT_PREFIX=${PREF}r \
      python3 probe_dump_m1.py >> $LOG 2>&1
    echo $PW | sudo -S -p X env PYTHONIOENCODING=utf-8 M1_WEIGHTS_DIR=$WD M1_CALIB_OUT=${SCAL%.json}r.json \
      python3 calibrate_any_m1.py >> $LOG 2>&1
    echo $PW | sudo -S -p X env PYTHONIOENCODING=utf-8 CALIB_COL_PAIRS_DIR=/home/uisrc/j1 \
      CALIB_COL_OUT=${COL%.json}r.json CALIB_COL_PREFIX=${PREF}r \
      CALIB_COL_LAYERS=conv1_2,conv2_1,conv2_2,conv3_1,conv3_2,fc1,fc2 CALIB_COL_SCALAR=${SCAL%.json}r.json \
      python3 calibrate_col.py >> $LOG 2>&1
    echo $PW | sudo -S -p X env PYTHONIOENCODING=utf-8 M1_WEIGHTS_DIR=$WD M1_OFFSET=$OFF M1_LIMIT=$LIMIT M1_BATCH=1 \
      M1_CALIB=${SCAL%.json}r.json M1_CALIB_COL=${COL%.json}r.json \
      M1_LOGITS_OUT=/home/uisrc/j1/logits_${PREF}r_off$OFF.npy \
      python3 run_m1_gazelle.py >> $LOG 2>&1
    T2=$(date +%s)
    ACC=$(grep -a 'FINAL:' $LOG | tail -1 | grep -aoE '[0-9]+[.][0-9]+' | tail -1)
    [ -z "$ACC" ] && ACC=-1
  fi
  RUNMIN=$(( (T2-T1)/60 ))
  COOL=360; LOW=0
  if awk "BEGIN{exit !($ACC < $EXPECT - 5)}"; then
    LOW=1; COOL=1800
    log "!!! ALERT [M1 seg#$N] off=$OFF acc=$ACC < expect-5 (<=$(awk "BEGIN{printf \"%.1f\", $EXPECT-5}")) -> cooldown ${COOL}s"
    echo "ALERT [M1 seg#$N] off=$OFF acc=$ACC" >> $STATE
  fi
  log "=== [M1 seg#$N] off=$OFF acc=$ACC (expect $EXPECT) calib=${CMIN}min run=${RUNMIN}min limit=$LIMIT ==="
  state "M1 seg#$N off=$OFF acc=$ACC calib=${CMIN}min run=${RUNMIN}min limit=$LIMIT low=$LOW NOW=$(date '+%F %T UTC')"
  log "=== [M1 seg#$N] cooldown ${COOL}s ==="
  sleep $COOL
}

log "==================== CICC M1 PARTIAL CONTROLLED START ===================="
state "BOOT $(date '+%F %T')"

# --- M1 抽样 [0:50], 5 段 × 10 张 ---
SEGCNT=0
for OFF in $(seq 0 10 40); do
  SEGCNT=$((SEGCNT+1))
  run_seg $OFF $SEGCNT 10
  [ $((SEGCNT % 3)) -eq 0 ] && {
    log "=== mid-run chip compass_cali ==="
    echo $PW | sudo -S -p X timeout 1100 compass_cali >> $LOG 2>&1
    state "CHIP-CALI seg#$SEGCNT $(date '+%F %T')"
    log "=== chip cali cooldown 6min ==="
    sleep 360
  }
done
log "==================== M1 PARTIAL DONE ===================="
state "M1-DONE $(date '+%F %T')"
