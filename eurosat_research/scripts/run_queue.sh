#!/bin/bash
# run_queue.sh — 在容器内顺序跑一批实验 (GPU 编号作为参数)
# 从 repo 根运行, 保证 data/EuroSAT_RGB 相对路径解析正确
REPO=/workspace/Ltsimulator-test
cd $REPO
GPU=$1
shift
PY=/local/miniconda/envs/moca_llm/bin/python
for c in "$@"; do
  echo "[GPU$GPU] $(date) starting $c"
  $PY auto_research/src/runner.py --config auto_research/configs/$c.json --gpu $GPU --base-dir $REPO/auto_research/runs > auto_research/logs/$c.log 2>&1
  echo "[GPU$GPU] $(date) finished $c (exit=$?)"
done
echo "[GPU$GPU] ALL DONE"
