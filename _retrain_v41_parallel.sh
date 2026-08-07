#!/bin/bash
# ================================================================================
# v4.1 并行通道 — 单任务运行 + 完整结果记录 (与 _retrain_v41_runner.sh 同构)
#
#  用法: bash _retrain_v41_parallel.sh <job>     job ∈ {m2, m3}
#  与顺序 runner 并行时: 本通道完成的 job 会写 .done → 顺序 runner 到时自动 SKIP
# ================================================================================
cd /mnt/e/LT-Simulator/train-test || exit 99
mkdir -p logs
STATE=logs/retrain_v41_state.txt
touch "$STATE"

declare -A JOBS
JOBS[m2]="src/training/model2_spacenet_v1_phase4_v3.py"
JOBS[m3]="src/training/model3_spacenet_v2_phase4_v3.py"

log_state() { echo "$(date '+%F %T') $*" >> "$STATE"; }

run() {
  local name="$1"; local cmdline="$2"
  if [ -f "logs/retrain_v41_${name}.done" ]; then
    echo "===== SKIP $name (already done: $(cat logs/retrain_v41_${name}.done)) ====="
    return 0
  fi
  echo "===== [$(date '+%F %T')] START $name (parallel lane) ====="
  log_state "START $name (parallel lane)"
  cmd.exe /c "set PYTHONIOENCODING=utf-8&& set PYTHONUTF8=1&& python -u $cmdline > logs/retrain_v41_${name}.log 2>&1"
  local rc=$?
  if [ $rc -eq 0 ]; then
    local best=$(grep -oE "Int8 最佳 val: +[0-9.]+%" "logs/retrain_v41_${name}.log" | tail -1)
    local final=$(grep -oE "Int8 模式[^:]*准确率: +[0-9.]+%" "logs/retrain_v41_${name}.log" | tail -1)
    local testq=$(grep -oE "Int8 模式 test 准确率[^:]*: +[0-9.]+%" "logs/retrain_v41_${name}.log" | tail -1)
    [ -z "$best" ]  && best=$(grep -oE "Int8 最佳准确率: +[0-9.]+%" "logs/retrain_v41_${name}.log" | tail -1)
    echo "DONE $(date '+%F %T') | ${best:-val:?} | ${final:-final:?} | ${testq:-}" > "logs/retrain_v41_${name}.done"
    log_state "DONE $name (parallel) exit=0 | $best | $final | $testq"
    echo "===== [$(date '+%F %T')] END $name exit=0 | $best | $final | $testq ====="
  else
    log_state "FAIL $name (parallel) exit=$rc"
    echo "===== [$(date '+%F %T')] END $name exit=$rc (详见 logs/retrain_v41_${name}.log) ====="
  fi
  return $rc
}

job="$1"
if [ -z "$job" ] || [ -z "${JOBS[$job]}" ]; then
  echo "用法: bash _retrain_v41_parallel.sh <job>, job ∈ {m2, m3}"
  exit 1
fi
run "$job" "${JOBS[$job]}"
echo "===== [$(date '+%F %T')] PARALLEL LANE $job FINISHED ====="
