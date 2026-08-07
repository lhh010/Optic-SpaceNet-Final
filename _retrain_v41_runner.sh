#!/bin/bash
# ================================================================================
# v4.1 重训流水线 — 可断点续跑 (新会话无缝衔接)
#
#  用法 (任何会话, 在 WSL 下):
#    cd /mnt/e/LT-Simulator/train-test && bash _retrain_v41_runner.sh
#   (建议后台运行: bash _retrain_v41_runner.sh > logs/retrain_v41_pipeline.log 2>&1 &)
#
#  续跑原理:
#    - 状态文件 logs/retrain_v41_state.txt: 每个任务 START/DONE/FAIL 时间戳与结果
#    - 完成标记 logs/retrain_v41_<job>.done: 内容 = 完成时间 + 提取的 val 精度
#    - 已 done 的任务自动 SKIP → 新会话重跑本脚本即可接着未完任务继续
#    - 任务中途被杀 → 无 .done 标记 → 下次自动重跑该任务
#    - ★ 任务内断点续训: M1/M2/M3 脚本每 5 epoch 存 <权重>.ckpt (model+optimizer+
#      scheduler+b est_acc), 重跑时自动从 checkpoint 继续 (无需重头训练) — 见脚本内
#      "[resume] 从 checkpoint 恢复" 打印; 训练成功完成后脚本自动删除 .ckpt
#  编码: 经 cmd.exe 传递 PYTHONIOENCODING=utf-8 (WSL interop 不传环境变量给 Windows exe)
# ================================================================================
cd /mnt/e/LT-Simulator/train-test || exit 99
mkdir -p logs
STATE=logs/retrain_v41_state.txt
touch "$STATE"

declare -A JOBS
JOBS[m4]="src/training/model4_minivgg_gap_phase4_v3.py"
JOBS[m1a]="src/training/model1_baseline_phase4_v3.py --variant A"
JOBS[m1b]="src/training/model1_baseline_phase4_v3.py --variant B"
JOBS[m2]="src/training/model2_spacenet_v1_phase4_v3.py"
JOBS[m3]="src/training/model3_spacenet_v2_phase4_v3.py"

log_state() { echo "$(date '+%F %T') $*" >> "$STATE"; }

run() {
  local name="$1"; local cmdline="$2"
  if [ -f "logs/retrain_v41_${name}.done" ]; then
    echo "===== SKIP $name (already done: $(cat logs/retrain_v41_${name}.done)) ====="
    return 0
  fi
  echo "===== [$(date '+%F %T')] START $name ====="
  log_state "START $name"
  cmd.exe /c "set PYTHONIOENCODING=utf-8&& set PYTHONUTF8=1&& python -u $cmdline > logs/retrain_v41_${name}.log 2>&1"
  local rc=$?
  if [ $rc -eq 0 ]; then
    local best=$(grep -oE "Int8 最佳 val: +[0-9.]+%" "logs/retrain_v41_${name}.log" | tail -1)
    local final=$(grep -oE "Int8 模式[^:]*准确率: +[0-9.]+%" "logs/retrain_v41_${name}.log" | tail -1)
    local testq=$(grep -oE "Int8 模式 test 准确率[^:]*: +[0-9.]+%" "logs/retrain_v41_${name}.log" | tail -1)
    [ -z "$best" ]  && best=$(grep -oE "Int8 最佳准确率: +[0-9.]+%" "logs/retrain_v41_${name}.log" | tail -1)
    echo "DONE $(date '+%F %T') | ${best:-val:?} | ${final:-final:?} | ${testq:-}" > "logs/retrain_v41_${name}.done"
    log_state "DONE $name exit=0 | $best | $final | $testq"
    echo "===== [$(date '+%F %T')] END $name exit=0 | $best | $final | $testq ====="
  else
    log_state "FAIL $name exit=$rc"
    echo "===== [$(date '+%F %T')] END $name exit=$rc (详见 logs/retrain_v41_${name}.log) ====="
  fi
  return $rc
}

echo "========== 当前流水线状态 (logs/retrain_v41_state.txt) =========="
cat "$STATE" 2>/dev/null || echo "(空)"
echo "=================================================================="

run m4  "${JOBS[m4]}"
run m1a "${JOBS[m1a]}"
run m1b "${JOBS[m1b]}"
run m2  "${JOBS[m2]}"
run m3  "${JOBS[m3]}"

echo "===== [$(date '+%F %T')] ALL JOBS FINISHED ====="
