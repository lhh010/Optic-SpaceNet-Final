#!/bin/bash
# M9 补段 off=1200 (labels 修复后补跑, 2026-08-15)
cd /home/uisrc/j1
LOG=/home/uisrc/m910_ccic.log
STATE=/home/uisrc/j1/ccic_state.txt
PW=5182
WD=/home/uisrc/j1/weights_m9_5400
SCAL=/home/uisrc/j1/calib_scalar_m9ccic.json
COL=/home/uisrc/j1/calib_col_m9ccic.json
PREF=probe_m9patch_

log(){ echo "[$(date '+%F %T')] $*" >> $LOG; }
state(){ echo "$*" > $STATE; }

T0=$(date +%s)
log "=== [M9-PATCH] off=1200 probe+calib start ==="
echo $PW | sudo -S -p X env DS3_WEIGHTS_DIR=$WD PROBE_OUT_PREFIX=$PREF python3 probe_dump_ds3.py >> $LOG 2>&1
echo $PW | sudo -S -p X env DS3_WEIGHTS_DIR=$WD DS3_CALIB_OUT=$SCAL python3 calibrate_any_ds3.py >> $LOG 2>&1
echo $PW | sudo -S -p X env CALIB_COL_PAIRS_DIR=/home/uisrc/j1 CALIB_COL_OUT=$COL \
  CALIB_COL_PREFIX=$PREF CALIB_COL_LAYERS=s1a,s1ds,s2a,s2b,s2ds,s3a,s3b \
  CALIB_COL_SCALAR=$SCAL python3 calibrate_col.py >> $LOG 2>&1
T1=$(date +%s); CMIN=$(( (T1-T0)/60 )); [ $CMIN -lt 1 ] && CMIN=1
RMAX=$(( 30 - CMIN )); [ $RMAX -gt 20 ] && RMAX=20; [ $RMAX -lt 8 ] && RMAX=8
LIMIT=$(( RMAX * 21 / 20 * 20 )); [ $LIMIT -lt 40 ] && LIMIT=40
log "=== [M9-PATCH] off=1200 calib ${CMIN}min -> run budget ${RMAX}min -> LIMIT=$LIMIT ==="
echo $PW | sudo -S -p X env DS3_WEIGHTS_DIR=$WD DS3_OFFSET=1200 DS3_LIMIT=$LIMIT DS3_BATCH=8 \
  DS3_CALIB=$SCAL DS3_CALIB_COL=$COL DS3_LOGITS_OUT=/home/uisrc/j1/logits_${PREF}_off1200.npy \
  python3 run_ds3_gazelle.py >> $LOG 2>&1
ACC=$(grep -a 'FINAL:' $LOG | tail -1 | grep -aoE '[0-9]+[.][0-9]+' | tail -1)
[ -z "$ACC" ] && ACC=-1
log "=== [M9-PATCH] off=1200 acc=$ACC calib=${CMIN}min budget=${RMAX}min limit=$LIMIT ==="
state "M9-PATCH off=1200 acc=$ACC $(date '+%F %T')"
log "=== [M9-PATCH] cooldown 360s ==="
sleep 360
echo "M9-PATCH DONE acc=$ACC"
