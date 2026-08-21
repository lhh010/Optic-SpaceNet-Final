#!/bin/bash
# ==============================================================================
# CICC1003564 | M9/M10 全量 5400 验证受控驱动 (2026-08-15)
# 规则 (用户 2026-08-15 指示, 硬性):
#   1. 单次调用 (校准+跑批) 尽量 <=30min; 校准耗时从 30min 预算扣除 → 跑批 = 30 - calib_min
#   2. 两次调用之间 >5min 空余 (正常冷却 6min)
#   3. 校准后不达标 (EBR<8 / error_std 恶化) → 缩减跑批 (LIMIT 按预算折算)
#   4. acc < expect-5pt → 冷却升级 15-30min + ALERT 标记 (及时汇报)
#   5. 0% 段自愈: 重 probe+calib 再跑一次
#   6. 每 3 段 fresh chip-level compass_cali + EBR 复查 (热崩溃预防)
#   7. 每段后写状态文件 (轮询汇报用)
# 用法: sudo 不需要, 以 uisrc 运行 (脚本内部 sudo)
# ==============================================================================
cd /home/uisrc/j1
LOG=/home/uisrc/m910_ccic.log
STATE=/home/uisrc/j1/ccic_state.txt
PW=5182
M9_EXPECT=94.9
M10_EXPECT=96.4
M9_RATE=21          # 张/min (400张/19min 实测)
M10_RATE=17          # 张/min (340张/18min 实测, 整数避免bash算术错误)

log(){ echo "[$(date '+%F %T')] $*" >> $LOG; }
state(){ echo "$*" > $STATE; }

chip_cali(){
  log "=== chip-level compass_cali start ==="
  echo $PW | sudo -S -p X bash -c 'cd /home/uisrc/sample_code/code && timeout 1100 python3 calibrate_sample.py' >> $LOG 2>&1
  log "=== compass_cali done ==="
}

ebr_check(){
  # returns first EBR value parsed from evb_test 'ebr: [...]' line
  echo $PW | sudo -S -p X timeout 600 python3 /home/uisrc/sample_code/code/evb_test_sample.py > /tmp/evb_out.txt 2>&1
  cat /tmp/evb_out.txt >> $LOG
  grep -a 'ebr:' /tmp/evb_out.txt | grep -aoE '[0-9]+[.][0-9]+' | head -1
}

canary(){
  log "=== MNIST canary 1000 (lsqplus) ==="
  echo $PW | sudo -S -p X timeout 900 env MNIST_METHOD=lsqplus MNIST_LIMIT=1000 MNIST_BATCH=50 MNIST_MODE=scale \
    python3 /home/uisrc/mnist/run_mnist_gazelle.py >> $LOG 2>&1
  grep -aE 'FINAL|acc|gap' $LOG | tail -3
}

mini_run(){
  # EuroSAT 200 mini-run (开窗判据之一), 用传入 calib
  local WD=$1 SCAL=$2 COL=$3
  log "=== mini-run 200 (EuroSAT 开窗判据) ==="
  echo $PW | sudo -S -p X env DS3_WEIGHTS_DIR=$WD DS3_OFFSET=0 DS3_LIMIT=200 DS3_BATCH=8 \
    DS3_CALIB=$SCAL DS3_CALIB_COL=$COL python3 run_ds3_gazelle.py >> $LOG 2>&1
  grep -a 'FINAL:' $LOG | tail -1
}

run_seg(){
  # $1=WD $2=PREF $3=SCAL $4=COL $5=OFF $6=EXPECT $7=TAG
  local WD=$1 PREF=$2 SCAL=$3 COL=$4 OFF=$5 EXPECT=$6 TAG=$7
  local RATE; [ "$TAG" = "M9" ] && RATE=$M9_RATE || RATE=$M10_RATE
  T0=$(date +%s)
  log "=== [$TAG] off=$OFF probe+calib start ==="
  echo $PW | sudo -S -p X env PYTHONIOENCODING=utf-8 DS3_WEIGHTS_DIR=$WD PROBE_OUT_PREFIX=${PREF} python3 probe_dump_ds3.py >> $LOG 2>&1
  echo $PW | sudo -S -p X env PYTHONIOENCODING=utf-8 DS3_WEIGHTS_DIR=$WD DS3_CALIB_OUT=$SCAL python3 calibrate_any_ds3.py >> $LOG 2>&1
  echo $PW | sudo -S -p X env PYTHONIOENCODING=utf-8 CALIB_COL_PAIRS_DIR=/home/uisrc/j1 CALIB_COL_OUT=$COL \
    CALIB_COL_PREFIX=$PREF CALIB_COL_LAYERS=s1a,s1ds,s2a,s2b,s2ds,s3a,s3b \
    CALIB_COL_SCALAR=$SCAL python3 calibrate_col.py >> $LOG 2>&1
  T1=$(date +%s); CMIN=$(( (T1-T0)/60 )); [ $CMIN -lt 1 ] && CMIN=1
  RMAX=$(( 30 - CMIN )); [ $RMAX -gt 20 ] && RMAX=20; [ $RMAX -lt 8 ] && RMAX=8
  # LIMIT = budget * rate, 向下取整到 20 的倍数 (最后一段由 n_test 自动截断)
  RAW=$(( RMAX * RATE ))
  LIMIT=$(( RAW / 20 * 20 ))
  [ -z "$LIMIT" ] && LIMIT=340
  [ $LIMIT -lt 40 ] && LIMIT=40
  LAST_LIMIT=$LIMIT
  log "=== [$TAG] off=$OFF calib ${CMIN}min -> run budget ${RMAX}min -> LIMIT=$LIMIT ==="
  echo $PW | sudo -S -p X env PYTHONIOENCODING=utf-8 DS3_WEIGHTS_DIR=$WD DS3_OFFSET=$OFF DS3_LIMIT=$LIMIT DS3_BATCH=8 \
    DS3_CALIB=$SCAL DS3_CALIB_COL=$COL DS3_LOGITS_OUT=/home/uisrc/j1/logits_${PREF}_off$OFF.npy \
    python3 run_ds3_gazelle.py >> $LOG 2>&1
  ACC=$(grep -a 'FINAL:' $LOG | tail -1 | grep -aoE '[0-9]+[.][0-9]+' | tail -1)
  [ -z "$ACC" ] && ACC=-1
  # 0% 自愈: 重 probe+calib, 重跑一次
  if awk "BEGIN{exit !($ACC == 0.00)}"; then
    log "!!! [$TAG] off=$OFF 0% -> self-heal re-probe+calib ==="
    echo $PW | sudo -S -p X env DS3_WEIGHTS_DIR=$WD PROBE_OUT_PREFIX=${PREF}r python3 probe_dump_ds3.py >> $LOG 2>&1
    echo $PW | sudo -S -p X env DS3_WEIGHTS_DIR=$WD DS3_CALIB_OUT=${SCAL%.json}r.json python3 calibrate_any_ds3.py >> $LOG 2>&1
    echo $PW | sudo -S -p X env PYTHONIOENCODING=utf-8 CALIB_COL_PAIRS_DIR=/home/uisrc/j1 CALIB_COL_OUT=${COL%.json}r.json \
      CALIB_COL_PREFIX=${PREF}r CALIB_COL_LAYERS=s1a,s1ds,s2a,s2b,s2ds,s3a,s3b \
      CALIB_COL_SCALAR=${SCAL%.json}r.json python3 calibrate_col.py >> $LOG 2>&1
    echo $PW | sudo -S -p X env PYTHONIOENCODING=utf-8 DS3_WEIGHTS_DIR=$WD DS3_OFFSET=$OFF DS3_LIMIT=$LIMIT DS3_BATCH=8 \
      DS3_CALIB=${SCAL%.json}r.json DS3_CALIB_COL=${COL%.json}r.json \
      DS3_LOGITS_OUT=/home/uisrc/j1/logits_${PREF}r_off$OFF.npy \
      python3 run_ds3_gazelle.py >> $LOG 2>&1
    ACC=$(grep -a 'FINAL:' $LOG | tail -1 | grep -aoE '[0-9]+[.][0-9]+' | tail -1)
    [ -z "$ACC" ] && ACC=-1
  fi
  # 降准 >5pt → 冷却升级
  COOL=360
  LOW=0
  if awk "BEGIN{exit !($ACC < $EXPECT - 5)}"; then
    LOW=1; COOL=1800
    log "!!! ALERT [$TAG] off=$OFF acc=$ACC < expect-5 (<=$(awk "BEGIN{printf \"%.1f\", $EXPECT-5}")) -> cooldown ${COOL}s"
    echo "ALERT [$TAG] off=$OFF acc=$ACC < expect-5" >> $STATE
  fi
  log "=== [$TAG] off=$OFF acc=$ACC (expect $EXPECT) calib=${CMIN}min budget=${RMAX}min limit=$LIMIT ==="
  state "$TAG off=$OFF acc=$ACC calib=${CMIN}min budget=${RMAX}min limit=$LIMIT low=$LOW $(date '+%F %T')"
  log "=== [$TAG] cooldown ${COOL}s ==="
  sleep $COOL
}

# ============================== 主流程 ==============================
log "==================== CICC M9/M10 FULL CONTROLLED START ===================="
state "BOOT $(date '+%F %T')"

# --- 开窗: 侦测 + fresh compass_cali + 四项判据 ---
log "=== open-window: occupancy check ==="
who >> $LOG 2>&1
ps aux | grep -iE 'gazelle|compass|server' | grep -v grep >> $LOG 2>&1

if [ "$SKIP_OPEN_CALI" != "1" ]; then
  chip_cali
  EBR1=$(ebr_check); log "EBR after cali: $EBR1"
  if [ -z "$EBR1" ] || awk "BEGIN{exit !($EBR1 < 8)}"; then
    log "!!! EBR low ($EBR1), retry compass_cali once"
    chip_cali
    EBR1=$(ebr_check); log "EBR after retry: $EBR1"
    if [ -z "$EBR1" ] || awk "BEGIN{exit !($EBR1 < 8)}"; then
      log "!!! EBR still <8 -> ABORT"; state "ABORT EBR=$EBR1"
      exit 1
    fi
  fi
  state "WINDOW-OPEN EBR=$EBR1 $(date '+%F %T')"
  canary
  state "CANARY-DONE EBR=$EBR1 $(date '+%F %T')"
else
  log "SKIP_OPEN_CALI=1: reuse existing fresh cali (done manually 2026-08-15)"
  state "WINDOW-OPEN (manual cali) EBR=9.61/9.67 canary=-0.20pt $(date '+%F %T')"
fi

# 每 3 段 fresh chip compass_cali + EBR 复查 (热崩溃预防)
mid_check(){
  local N=$1
  chip_cali
  EBRM=$(ebr_check); log "EBR mid-run check (seg#$N): $EBRM"
  state "CHIP-CALI seg#$N EBR=$EBRM $(date '+%F %T')"
  log "=== chip cali cooldown 6min ==="
  sleep 360
}

# --- M9 剩余 [1200:5400] (M9_START 支持断点续跑; SKIP_M9=1 跳过) ---
SCAL_M9=/home/uisrc/j1/calib_scalar_m9ccic.json
COL_M9=/home/uisrc/j1/calib_col_m9ccic.json
SEGCNT=${SEGCNT:-0}
if [ "$SKIP_M9" != "1" ]; then
for OFF in $(seq 1200 400 5200); do
  if [ "$M9_START" != "" ] && [ $OFF -lt $M9_START ]; then continue; fi
  run_seg /home/uisrc/j1/weights_m9_5400 probe_m9ccic_ $SCAL_M9 $COL_M9 $OFF $M9_EXPECT M9
  SEGCNT=$((SEGCNT+1))
  [ $((SEGCNT % 3)) -eq 0 ] && mid_check $SEGCNT
done
log "==================== M9 FULL DONE ===================="
state "M9-DONE $(date '+%F %T')"
else
  log "SKIP_M9=1: M9 segments already done (2026-08-15)"
  state "M9-DONE(skip) $(date '+%F %T')"
fi

# --- M10 全量 [0:5400] (动态 OFFSET 累加 LIMIT, 避免漏样本; M10_START 断点续跑) ---
SCAL_M10=/home/uisrc/j1/calib_scalar_m10ccic.json
COL_M10=/home/uisrc/j1/calib_col_m10ccic.json
OFF=${M10_START:-0}
LAST_LIMIT=340
while [ $OFF -lt 5400 ]; do
  run_seg /home/uisrc/j1/weights_m10_5400 probe_m10ccic_ $SCAL_M10 $COL_M10 $OFF $M10_EXPECT M10
  [ -z "$LAST_LIMIT" ] && LAST_LIMIT=340
  OFF=$(( OFF + LAST_LIMIT ))
  SEGCNT=$((SEGCNT+1))
  [ $((SEGCNT % 3)) -eq 0 ] && mid_check $SEGCNT
done
log "==================== M10 FULL DONE ===================="
state "ALL-DONE $(date '+%F %T')"
log "==================== CICC M9/M10 FULL CONTROLLED ALL DONE ===================="
