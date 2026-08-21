#!/bin/bash
# ==============================================================================
# CICC1003564 | M7 (J1-w075, C0=12) 全量 5400 真机验证受控驱动 (2026-08-16)
# 调度策略 (与 M9/M10 相同, 用户 2026-08-15 指示, 硬性):
#   1. 单次调用 (校准+跑批) <=30min; 跑批 = 30 - calib_min
#   2. 两次调用之间 >5min 空余 (冷却 6min)
#   3. 校准后不达标 → 缩减跑批
#   4. acc < expect-5pt → 冷却升级 30min + ALERT (及时汇报)
#   5. 0% 段自愈重采
#   6. 每 3 段 fresh chip compass_cali + EBR 复查
#   7. 每段写 state 文件 (汇报: 批次耗时/准确率/当前时间)
# 部署链: run_j1_gazelle.py (J1 env) + probe_dump.py (PROBE_ROWS) +
#         calibrate_any.py (J1_CALIB_OUT) + calibrate_col.py (s1a,s2a,s2b,s3a,s3b)
# ==============================================================================
cd /home/uisrc/j1
LOG=/home/uisrc/m7_ccic.log
STATE=/home/uisrc/j1/m7_state.txt
PW=5182
EXPECT=${M7_EXPECT:-95.0}     # v8 软件 test 95.00
RATE=30                       # 张/min (M7 5 光层估 ~1.6s/张)
WD=/home/uisrc/j1/weights_m7_5400
SCAL=/home/uisrc/j1/calib_scalar_m7ccic.json
COL=/home/uisrc/j1/calib_col_m7ccic.json
PREF=probe_m7ccic_

log(){ echo "[$(date '+%F %T')] $*" >> $LOG; }
state(){ echo "$*" > $STATE; }

chip_cali(){
  log "=== chip-level compass_cali start ==="
  echo $PW | sudo -S -p X bash -c 'cd /home/uisrc/sample_code/code && timeout 1100 compass_cali' >> $LOG 2>&1
  log "=== compass_cali done ==="
}

ebr_check(){
  echo $PW | sudo -S -p X timeout 600 python3 /home/uisrc/sample_code/code/evb_test_sample.py > /tmp/evb_m7.txt 2>&1
  cat /tmp/evb_m7.txt >> $LOG
  grep -a 'ebr:' /tmp/evb_m7.txt | grep -aoE '[0-9]+[.][0-9]+' | head -1
}

canary(){
  log "=== MNIST canary 1000 (lsqplus) ==="
  echo $PW | sudo -S -p X timeout 900 env MNIST_METHOD=lsqplus MNIST_LIMIT=1000 MNIST_BATCH=50 MNIST_MODE=scale \
    python3 /home/uisrc/mnist/run_mnist_gazelle.py >> $LOG 2>&1
  grep -aE 'FINAL HW|reference|gap' $LOG | tail -3
}

run_seg(){
  # $1=OFF $2=SEGN (段序号, 汇报用)
  local OFF=$1 N=$2
  T0=$(date +%s)
  log "=== [M7 seg#$N] off=$OFF probe+calib start ==="
  echo $PW | sudo -S -p X env PYTHONIOENCODING=utf-8 J1_WEIGHTS_DIR=$WD PROBE_ROWS=30000 \
    python3 probe_dump.py >> $LOG 2>&1
  echo $PW | sudo -S -p X env PYTHONIOENCODING=utf-8 J1_WEIGHTS_DIR=$WD J1_CALIB_OUT=$SCAL \
    python3 calibrate_any.py >> $LOG 2>&1
  echo $PW | sudo -S -p X env PYTHONIOENCODING=utf-8 CALIB_COL_PAIRS_DIR=/home/uisrc/j1 \
    CALIB_COL_OUT=$COL CALIB_COL_LAYERS=s1a,s2a,s2b,s3a,s3b CALIB_COL_SCALAR=$SCAL \
    python3 calibrate_col.py >> $LOG 2>&1
  T1=$(date +%s); CMIN=$(( (T1-T0)/60 )); [ $CMIN -lt 1 ] && CMIN=1
  RMAX=$(( 30 - CMIN )); [ $RMAX -gt 20 ] && RMAX=20; [ $RMAX -lt 8 ] && RMAX=8
  RAW=$(( RMAX * RATE ))
  LIMIT=$(( RAW / 20 * 20 )); [ -z "$LIMIT" ] && LIMIT=300
  [ $LIMIT -lt 40 ] && LIMIT=40
  LAST_LIMIT=$LIMIT
  log "=== [M7 seg#$N] off=$OFF calib ${CMIN}min -> run budget ${RMAX}min -> LIMIT=$LIMIT ==="
  echo $PW | sudo -S -p X env PYTHONIOENCODING=utf-8 J1_WEIGHTS_DIR=$WD J1_OFFSET=$OFF J1_LIMIT=$LIMIT J1_BATCH=8 \
    J1_CALIB=$SCAL J1_CALIB_COL=$COL J1_LOGITS_OUT=/home/uisrc/j1/logits_${PREF}_off$OFF.npy \
    python3 run_j1_gazelle.py >> $LOG 2>&1
  T2=$(date +%s)
  ACC=$(grep -a 'FINAL:' $LOG | tail -1 | grep -aoE '[0-9]+[.][0-9]+' | tail -1)
  [ -z "$ACC" ] && ACC=-1
  if awk "BEGIN{exit !($ACC == 0.00)}"; then
    log "!!! [M7 seg#$N] off=$OFF 0% -> self-heal re-probe+calib ==="
    echo $PW | sudo -S -p X env PYTHONIOENCODING=utf-8 J1_WEIGHTS_DIR=$WD PROBE_ROWS=30000 \
      python3 probe_dump.py >> $LOG 2>&1
    echo $PW | sudo -S -p X env PYTHONIOENCODING=utf-8 J1_WEIGHTS_DIR=$WD J1_CALIB_OUT=${SCAL%.json}r.json \
      python3 calibrate_any.py >> $LOG 2>&1
    echo $PW | sudo -S -p X env PYTHONIOENCODING=utf-8 CALIB_COL_PAIRS_DIR=/home/uisrc/j1 \
      CALIB_COL_OUT=${COL%.json}r.json CALIB_COL_LAYERS=s1a,s2a,s2b,s3a,s3b CALIB_COL_SCALAR=${SCAL%.json}r.json \
      python3 calibrate_col.py >> $LOG 2>&1
    echo $PW | sudo -S -p X env PYTHONIOENCODING=utf-8 J1_WEIGHTS_DIR=$WD J1_OFFSET=$OFF J1_LIMIT=$LIMIT J1_BATCH=8 \
      J1_CALIB=${SCAL%.json}r.json J1_CALIB_COL=${COL%.json}r.json \
      J1_LOGITS_OUT=/home/uisrc/j1/logits_${PREF}r_off$OFF.npy \
      python3 run_j1_gazelle.py >> $LOG 2>&1
    T2=$(date +%s)
    ACC=$(grep -a 'FINAL:' $LOG | tail -1 | grep -aoE '[0-9]+[.][0-9]+' | tail -1)
    [ -z "$ACC" ] && ACC=-1
  fi
  RUNMIN=$(( (T2-T1)/60 ))
  COOL=360; LOW=0
  if awk "BEGIN{exit !($ACC < $EXPECT - 5)}"; then
    LOW=1; COOL=1800
    log "!!! ALERT [M7 seg#$N] off=$OFF acc=$ACC < expect-5 (<=$(awk "BEGIN{printf \"%.1f\", $EXPECT-5}")) -> cooldown ${COOL}s"
    echo "ALERT [M7 seg#$N] off=$OFF acc=$ACC < expect-5" >> $STATE
  fi
  log "=== [M7 seg#$N] off=$OFF acc=$ACC (expect $EXPECT) calib=${CMIN}min run=${RUNMIN}min limit=$LIMIT ==="
  state "M7 seg#$N off=$OFF acc=$ACC calib=${CMIN}min run=${RUNMIN}min limit=$LIMIT low=$LOW NOW=$(date '+%F %T UTC')"
  log "=== [M7 seg#$N] cooldown ${COOL}s ==="
  sleep $COOL
}

# ============================== 主流程 ==============================
log "==================== CICC M7 FULL CONTROLLED START ===================="
state "BOOT $(date '+%F %T')"

if [ "$SKIP_OPEN_CALI" != "1" ]; then
  log "=== open-window: occupancy check ==="
  who >> $LOG 2>&1
  ps aux | grep -iE 'gazelle|compass|server' | grep -v grep >> $LOG 2>&1
  chip_cali
  EBR1=$(ebr_check); log "EBR after cali: $EBR1"
  if [ -z "$EBR1" ] || awk "BEGIN{exit !($EBR1 < 8)}"; then
    chip_cali
    EBR1=$(ebr_check); log "EBR after retry: $EBR1"
    if [ -z "$EBR1" ] || awk "BEGIN{exit !($EBR1 < 8)}"; then
      log "!!! EBR still <8 -> ABORT"; state "ABORT EBR=$EBR1"; exit 1
    fi
  fi
  canary
  state "WINDOW-OPEN EBR=$EBR1 $(date '+%F %T')"
else
  log "SKIP_OPEN_CALI=1: reuse existing fresh cali (manual 2026-08-16)"
  state "WINDOW-OPEN (manual) $(date '+%F %T')"
fi

# --- M7 全量 [0:5400], 动态 OFFSET ---
OFF=${M7_START:-0}
LAST_LIMIT=300
SEGCNT=0
while [ $OFF -lt 5400 ]; do
  SEGCNT=$((SEGCNT+1))
  run_seg $OFF $SEGCNT
  [ -z "$LAST_LIMIT" ] && LAST_LIMIT=300
  OFF=$(( OFF + LAST_LIMIT ))
  [ $((SEGCNT % 3)) -eq 0 ] && {
    chip_cali
    EBRM=$(ebr_check); log "EBR mid-run check (seg#$SEGCNT): $EBRM"
    state "CHIP-CALI seg#$SEGCNT EBR=$EBRM $(date '+%F %T')"
    log "=== chip cali cooldown 6min ==="
    sleep 360
  }
done
log "==================== M7 FULL DONE ===================="
state "M7-DONE $(date '+%F %T')"
