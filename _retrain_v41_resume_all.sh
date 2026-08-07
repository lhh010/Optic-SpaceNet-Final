#!/bin/bash
# ================================================================================
# v4.1 重训一键恢复 — 双通道并行启动 (web 重启 / 新会话后使用)
#
#  用法 (WSL):
#    cd /mnt/e/LT-Simulator/train-test && bash _retrain_v41_resume_all.sh
#
#  效果:
#    通道 A: bash _retrain_v41_runner.sh      — M1-B 续训 (自动 skip m4/m1a/m2)
#    通道 B: bash _retrain_v41_parallel.sh m3 — M3-KD 续训
#  幂等: 已 .done 的任务自动跳过; 未完成任务从 .ckpt 自动续 (最多丢 5 epochs)
# ================================================================================
cd /mnt/e/LT-Simulator/train-test || exit 99
mkdir -p logs
TS=$(date '+%m%d_%H%M%S')

echo "===== [$(date '+%F %T')] RESUME-ALL: 启动双通道 ====="
nohup bash _retrain_v41_runner.sh        > "logs/retrain_v41_pipeline_${TS}.log" 2>&1 &
echo "通道 A (runner: M1-B 等) PID=$!  log=logs/retrain_v41_pipeline_${TS}.log"
nohup bash _retrain_v41_parallel.sh m3   > "logs/retrain_v41_m3_lane_${TS}.log" 2>&1 &
echo "通道 B (parallel: M3-KD) PID=$!  log=logs/retrain_v41_m3_lane_${TS}.log"
echo "===== [$(date '+%F %T')] RESUME-ALL 已启动 (状态见 logs/retrain_v41_state.txt) ====="
