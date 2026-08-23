#!/bin/bash
# ==============================================================================
# Gazelle 真机校准 (M9/M10) — 在容器内运行。
# 流程: 连通性/工具链检查 → fresh compass_cali → probe → 标量 calib → 逐列 calib
#       → scp 拉回 calib json 到 /workspace/out/calib/
# 连接: ssh uisrc@192.168.31.158 (密码 5182)  板上目录 /home/uisrc/j1
# 注意: 校准与跑批必须同窗口背靠背; calib json 不可跨窗口复用 (stale −12.5pt)。
# ==============================================================================
set -uo pipefail
BOARD_HOST="${GAZELLE_HOST:-192.168.31.158}"
BOARD_USER="${GAZELLE_SSH_USER:-uisrc}"
BOARD_PASS="${GAZELLE_SSH_PASSWORD:-5182}"
BOARD_J1="${GAZELLE_BOARD_J1:-/home/uisrc/j1}"
MODEL="${1:-m10}"                      # m9 | m10
OUT_DIR="/workspace/out/calib"
TS="$(date +%Y%m%d_%H%M%S)"

case "$MODEL" in
  m9)  WD="weights_w075ds3"; PREF="probe_m9demo_";;
  m10) WD="weights_m10_5400"; PREF="probe_m10demo_";;
  *)   echo "用法: calib_board.sh [m9|m10]"; exit 2;;
esac

sshx() { SSHPASS="$BOARD_PASS" sshpass -e ssh -o StrictHostKeyChecking=no \
          -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 \
          "$BOARD_USER@$BOARD_HOST" "$@"; }
scpx() { SSHPASS="$BOARD_PASS" sshpass -e scp -o StrictHostKeyChecking=no \
          -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 "$@"; }

mkdir -p "$OUT_DIR"

echo "== [1/6] 板上连通性 =="
sshx "hostname; date -u" || { echo "!! 无法连接 $BOARD_USER@$BOARD_HOST"; exit 1; }

echo "== [2/6] 工具链检查 ($BOARD_J1) =="
MISS=""
for f in probe_dump_ds3.py calibrate_any_ds3.py calibrate_col.py run_ds3_gazelle.py; do
  sshx "test -f $BOARD_J1/$f" || MISS="$MISS $f"
done
sshx "test -d $BOARD_J1/$WD" || MISS="$MISS weights:$WD"
if [ -n "$MISS" ]; then
  echo "!! 板上缺失:$MISS — 自动从本地上传脚本 (脚本小, 权重包需已存在)"
  SRC=/workspace/app/eurosat_research/x0/scripts
  for f in probe_dump_ds3.py calibrate_any_ds3.py run_ds3_gazelle.py; do
    [ -f "$SRC/$f" ] && scpx "$SRC/$f" "$BOARD_USER@$BOARD_HOST:$BOARD_J1/$f" \
      && echo "  上传 $f OK"
  done
  [ -f /workspace/app/mnist/j1_board/calibrate_col.py ] && \
    scpx /workspace/app/mnist/j1_board/calibrate_col.py \
         "$BOARD_USER@$BOARD_HOST:$BOARD_J1/calibrate_col.py"
  sshx "test -d $BOARD_J1/$WD" || {
    echo "!! 板上缺权重包 $BOARD_J1/$WD (含 test_images_j1.npy 5400 张)。"
    echo "   请先上传权重包或确认该板已部署。中止。"; exit 1; }
fi
echo "   工具链 OK (weights=$WD)"

echo "== [3/6] 他人占用检查 =="
sshx "who; ps aux | grep -iE 'gazelle|compass|server' | grep -v grep | head -5; \
      (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -E ':(8000|22)' | head -3; \
      echo '--- BOARD_USAGE.md ---'; tail -20 /home/uisrc/BOARD_USAGE.md 2>/dev/null"
read -rp "确认无人占用且器件已冷却≥5min？输入 yes 继续: " CONFIRM
[ "$CONFIRM" = "yes" ] || { echo "!! 未确认占用/冷却状态，中止且不运行校准"; exit 1; }

echo "== [4/6] fresh compass_cali (约 10-18 分钟) =="
sshx "printf '%s\\n' '$BOARD_PASS' | sudo -S -p X timeout 1100 \
      compass_cali --mode-local > /tmp/cali_demo.log 2>&1; RC=\$?; \
      tail -20 /tmp/cali_demo.log; exit \$RC" || {
  echo "!! fresh compass_cali 失败，中止"; exit 1; }

echo "== [5/6] EBR 检查 =="
EVB=$(sshx "cd /home/uisrc/sample_code/code 2>/dev/null && \
  printf '%s\\n' '$BOARD_PASS' | sudo -S -p X timeout 600 python3 evb_test_sample.py 2>/dev/null | \
  tr '\r' '\n' | grep -aE 'error_std:|ebr:' | tail -2") || {
  echo "!! EVB 执行失败，中止"; exit 1; }
echo "   $EVB"
python3 - "$EVB" <<'PY' || { echo "!! EBR/error_std 未达严格判据，中止"; exit 1; }
import re
import sys

text = sys.argv[1]
def pair(key):
    m = re.search(r"%s\s*:\s*\[\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)" % key, text)
    return (float(m.group(1)), float(m.group(2))) if m else None

ebr = pair("ebr")
std = pair("error_std")
base = (4.694, 4.473)
if not ebr or not std or min(ebr) < 8:
    raise SystemExit(1)
if any((value - ref) / ref * 100.0 >= 2.0 for value, ref in zip(std, base)):
    raise SystemExit(1)
print("   PASS: EBR 两通道≥8，error_std 恶化<2%")
PY

echo "== [6/6] probe + 标量 calib + 逐列 calib =="
sshx "cd $BOARD_J1 && \
  printf '%s\\n' '$BOARD_PASS' | sudo -S -p X env PYTHONIOENCODING=utf-8 \
    DS3_WEIGHTS_DIR=$BOARD_J1/$WD PROBE_OUT_PREFIX=$PREF PROBE_IMAGES=512 \
    python3 probe_dump_ds3.py > /tmp/probe_demo.log 2>&1; \
  RC=\$?; echo probe_exit=\$RC; tail -8 /tmp/probe_demo.log | grep -aE 'resid_std|DONE'; exit \$RC" || {
  echo "!! probe 失败，中止"; exit 1; }
sshx "cd $BOARD_J1 && \
  printf '%s\\n' '$BOARD_PASS' | sudo -S -p X env PYTHONIOENCODING=utf-8 \
    DS3_WEIGHTS_DIR=$BOARD_J1/$WD DS3_CALIB_OUT=$BOARD_J1/calib_scalar_${MODEL}_demo_$TS.json \
    python3 calibrate_any_ds3.py > /tmp/calib_sc_demo.log 2>&1; \
  RC=\$?; echo sc_exit=\$RC; grep -a 'saved' /tmp/calib_sc_demo.log; SAVED=\$?; \
  [ \$RC -eq 0 ] && [ \$SAVED -eq 0 ]" || {
  echo "!! scalar calib 失败，中止"; exit 1; }
sshx "cd $BOARD_J1 && \
  printf '%s\\n' '$BOARD_PASS' | sudo -S -p X env PYTHONIOENCODING=utf-8 \
    CALIB_COL_PAIRS_DIR=$BOARD_J1 CALIB_COL_OUT=$BOARD_J1/calib_col_${MODEL}_demo_$TS.json \
    CALIB_COL_PREFIX=$PREF CALIB_COL_LAYERS=s1a,s1ds,s2a,s2b,s2ds,s3a,s3b \
    CALIB_COL_SCALAR=$BOARD_J1/calib_scalar_${MODEL}_demo_$TS.json \
    python3 calibrate_col.py > /tmp/calib_col_demo.log 2>&1; \
  RC=\$?; echo col_exit=\$RC; grep -a 'saved' /tmp/calib_col_demo.log; SAVED=\$?; \
  [ \$RC -eq 0 ] && [ \$SAVED -eq 0 ]" || {
  echo "!! column calib 失败，中止"; exit 1; }

echo "== 拉回 calib json =="
scpx "$BOARD_USER@$BOARD_HOST:$BOARD_J1/calib_scalar_${MODEL}_demo_$TS.json" "$OUT_DIR/" 2>/dev/null || {
  echo "!! scalar json 拉回失败，中止"; exit 1; }
scpx "$BOARD_USER@$BOARD_HOST:$BOARD_J1/calib_col_${MODEL}_demo_$TS.json" "$OUT_DIR/" 2>/dev/null || {
  echo "!! column json 拉回失败，中止"; exit 1; }
ls -la "$OUT_DIR" | tail -5
echo ""
echo "=========================================================="
echo " $MODEL 校准完成 (同窗口有效, 约 20 分钟内使用):"
echo "   标量: $OUT_DIR/calib_scalar_${MODEL}_demo_$TS.json"
echo "   逐列: $OUT_DIR/calib_col_${MODEL}_demo_$TS.json"
echo " 用法: run_sample_verify.py --calib-col $OUT_DIR/calib_col_${MODEL}_demo_$TS.json"
echo "       前端逐层: GAZELLE_CALIB_$([ $MODEL = m9 ] && echo 9 || echo 10)=$OUT_DIR/calib_col_${MODEL}_demo_$TS.json"
echo "       path-B 判据: GAZELLE_BOARD_CALIB_$([ $MODEL = m9 ] && echo 9 || echo 10)=calib_scalar_${MODEL}_demo_$TS.json"
echo "=========================================================="
