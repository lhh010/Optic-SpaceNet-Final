#!/usr/bin/env bash
# deploy.sh — 把光计算推理服务同步到 gazelle_sim 容器并重启常驻进程.
#
# 幂等: 重复执行 = 重新同步文件 + 杀掉旧进程 + 重启 + 健康检查.
# 用法: bash demo/deploy.sh   (在 repo 任意目录执行均可)
set -euo pipefail

REMOTE_HOST="fdusc-cpu-135"
CONTAINER="gazelle_sim"
REMOTE_DIR="/workspace/demo"
PYBIN="/local/miniconda/envs/moca_llm/bin/python"
PORT=8765
WEIGHT="spacenet_v2_phase4_v3_int8.pth"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[1/4] 打包 optic_server.py + optic_layers.py + 权重, 同步到 ${REMOTE_HOST}:${CONTAINER}:${REMOTE_DIR}"
tar -cf - \
  -C "${REPO_ROOT}/demo/remote" optic_server.py \
  -C "${REPO_ROOT}/src/core" optic_layers.py \
  -C "${REPO_ROOT}/weights" "${WEIGHT}" \
| ssh "${REMOTE_HOST}" "docker exec -i ${CONTAINER} sh -c 'mkdir -p ${REMOTE_DIR} && tar -x -C ${REMOTE_DIR}'"
echo "      同步完成."

echo "[2/4] 重启容器内常驻服务 (先杀旧进程, 幂等)"
ssh "${REMOTE_HOST}" "docker exec -i ${CONTAINER} sh -c '
  if [ -f ${REMOTE_DIR}/optic_server.pid ]; then
    kill \$(cat ${REMOTE_DIR}/optic_server.pid) 2>/dev/null || true
    sleep 1
  fi
  cd ${REMOTE_DIR}
  nohup ${PYBIN} optic_server.py --port ${PORT} > optic_server.log 2>&1 &
  echo \$! > optic_server.pid
  echo \"      pid=\$(cat optic_server.pid)\"'"

echo "[3/4] 等待 /health 就绪 (真机引擎加载需数十秒, 最多等 180s)"
ssh "${REMOTE_HOST}" "docker exec -i ${CONTAINER} sh -c '
  ok=0
  for i in \$(seq 1 60); do
    if ${PYBIN} -c \"import json,sys,urllib.request; d=json.load(urllib.request.urlopen(sys.argv[1],timeout=3)); print(d)\" \"http://127.0.0.1:${PORT}/health\" 2>/dev/null; then
      ok=1; break
    fi
    sleep 3
  done
  [ \"\$ok\" = 1 ] || { echo \"      [WARN] 服务未就绪, 日志如下:\"; tail -20 ${REMOTE_DIR}/optic_server.log; exit 1; }'"

echo "[4/4] 完成. 下一步 (本地):"
CIP="$(ssh "${REMOTE_HOST}" "docker inspect ${CONTAINER} --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'")"
echo "      1) 建立隧道:   ssh -N -L ${PORT}:${CIP}:${PORT} ${REMOTE_HOST}"
echo "         (容器 IP ${CIP}; 注意: 隧道目标是容器 IP, 不是 localhost — 8765 未发布到宿主机)"
echo "      2) 起本地后端: cd ${REPO_ROOT} && uvicorn demo.server.app:app --port 8000"
echo "      3) 打开浏览器: http://localhost:8000"
