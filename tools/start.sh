#!/bin/bash
# ==============================================================================
# Gazelle 决赛演示环境启动器 (本机)
# 功能: 1) 确保 Docker 环境 (构建镜像/启动容器, 挂载 代码/权重/数据/输出)
#       2) 菜单: [1] 校准  [2] 运行前端展示  [3] M9/M10 200 张抽样验证  [4] 退出
# 连接真机: ssh uisrc@192.168.31.158 (密码 5182); 板上 server_gazelle.py :8000
# ==============================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"                                  # Ltsimulator-EuroSAT-latest
OSIM_WEIGHTS="${OSIM_WEIGHTS:-$HOME/jichuangsai/osim/eurosat_research/weights}"
EUROSAT_DATA="${EUROSAT_DATA:-$HOME/jichuangsai/Ltsimulator-EuroSAT-3x3/data/EuroSAT_RGB}"
OUT_DIR="$REPO/tools/out"                                   # 校准/验证结果
IMAGE="gazelle-demo:1.0"
CONTAINER="gazelle-demo-run"
FRONT_PORT="${FRONT_PORT:-8000}"
OSIM_CONTAINER="${OSIM_CONTAINER:-LT-Simulator-container}"
OSIM_URL="${OPTIC_REMOTE_URL:-}"

mkdir -p "$OUT_DIR"

say()  { printf '\033[1;36m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
die()  { printf '\033[1;31m!! %s\033[0m\n' "$*"; exit 1; }

# ---------- 0. docker 可用性 ----------
command -v docker >/dev/null || die "未找到 docker, 请先安装/启动 docker"
docker info >/dev/null 2>&1 || die "docker daemon 未运行"

# ---------- 1. 镜像 ----------
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  say "首次使用: 构建镜像 $IMAGE (下载 torch CPU 约 5-15 分钟)..."
  docker build -t "$IMAGE" -f "$HERE/Dockerfile" "$HERE" || die "镜像构建失败"
  say "镜像构建完成"
else
  say "镜像 $IMAGE 已存在"
fi

# ---------- 2. 容器 ----------
if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
  say "启动容器 $CONTAINER (挂载代码/权重/数据/输出 + 串口设备)..."
  # 串口直连: 挂载本机 USB 转串口设备 (Gazelle console) + dialout 组
  DEV_ARGS=""
  for dev in /dev/ttyUSB* /dev/ttyACM*; do
    [ -e "$dev" ] && DEV_ARGS="$DEV_ARGS --device $dev"
  done
  [ -n "$DEV_ARGS" ] && say "挂载串口设备:$DEV_ARGS" || say "未发现串口设备 (USB 转串口线接入后需重建容器)"
  docker run -d --name "$CONTAINER" --restart unless-stopped \
    -p "$FRONT_PORT:8000" \
    --group-add dialout \
    $DEV_ARGS \
    -v "$REPO:/workspace/app" \
    -v "$OSIM_WEIGHTS:/workspace/weights:ro" \
    -v "$EUROSAT_DATA:/workspace/eurosat_data:ro" \
    -v "$OUT_DIR:/workspace/out" \
    -w /workspace/app \
    "$IMAGE" sleep infinity || die "容器启动失败"
else
  docker start "$CONTAINER" >/dev/null 2>&1 || true
fi
say "容器运行中 (前端端口 $FRONT_PORT, 结果输出 $OUT_DIR)"

# Model 3 固定走 osimulator。优先动态读取容器 IP，保留交接时地址作回退。
if [ -z "$OSIM_URL" ]; then
  OSIM_IP=$(docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$OSIM_CONTAINER" 2>/dev/null)
  OSIM_URL="http://${OSIM_IP:-172.17.0.3}:8765"
fi
say "Model 3 osimulator: $OSIM_URL"

# ---------- 3. 前端环境变量提示 ----------
cat <<'EOF'
  [真机调用环境变量] (容器内可通过 tools/*.sh 脚本传入, 或 export 后 exec)
    GAZELLE_HOST=192.168.31.158  GAZELLE_PORT=8000
    GAZELLE_WEIGHT_9=/workspace/weights/m9_j1w075ds3_v8probe15.pth
    GAZELLE_WEIGHT_10=/workspace/weights/m10_ds3pool3_v8probe15.pth
EOF

# ---------- 4. 菜单 ----------
while true; do
  echo ""
  echo "==================== Gazelle 演示环境 ===================="
  echo "  [1] 校准 (真机 compass_cali + probe + 逐列 calib, 约 25 分钟/模型)"
  echo "  [2] 运行前端展示 (uvicorn :$FRONT_PORT, 浏览器打开)"
  echo "  [3] M9/M10 200 张抽样验证并输出结果"
  echo "  [4] 连接/检查 Gazelle 硬件 (SSH 状态/启停 server / 串口直连)"
  echo "  [5] 退出"
  echo "=========================================================="
  read -rp "请选择 [1-4]: " CHOICE
  case "$CHOICE" in
    1)
      echo "  校准模型: [1] M9   [2] M10   [3] M9+M10"
      read -rp "  选择 [1-3]: " M
      case "$M" in
        1) docker exec "$CONTAINER" bash /workspace/app/tools/calib_board.sh m9 ;;
        2) docker exec "$CONTAINER" bash /workspace/app/tools/calib_board.sh m10 ;;
        3) docker exec "$CONTAINER" bash /workspace/app/tools/calib_board.sh m9
           docker exec "$CONTAINER" bash /workspace/app/tools/calib_board.sh m10 ;;
        *) echo "  无效选择";;
      esac
      ;;
    2)
      echo "  M9/M10 数据来源: [1] Gazelle 真机   [2] 离线 numpy 参考"
      echo "  Model 3 始终使用 osimulator ($OSIM_URL)"
      read -rp "  选择 [1-2]: " B
      ENVS="OPTIC_OSIM=1 OPTIC_REMOTE_URL=$OSIM_URL"
      ENVS="$ENVS GAZELLE_HOST=192.168.31.158 GAZELLE_PORT=8000"
      ENVS="$ENVS GAZELLE_WEIGHT_9=/workspace/weights/m9_j1w075ds3_v8probe15.pth"
      ENVS="$ENVS GAZELLE_WEIGHT_10=/workspace/weights/m10_ds3pool3_v8probe15.pth"
      # 新校准文件名带模型，避免把 M9 校准误用于 M10（或反之）。
      LATEST9=$(ls -t "$OUT_DIR/calib"/calib_col_m9_*.json 2>/dev/null | head -1)
      LATEST10=$(ls -t "$OUT_DIR/calib"/calib_col_m10_*.json 2>/dev/null | head -1)
      SCALAR9=$(ls -t "$OUT_DIR/calib"/calib_scalar_m9_*.json 2>/dev/null | head -1)
      SCALAR10=$(ls -t "$OUT_DIR/calib"/calib_scalar_m10_*.json 2>/dev/null | head -1)
      [ -n "$LATEST9" ] && ENVS="$ENVS GAZELLE_CALIB_9=/workspace/out/calib/$(basename "$LATEST9")"
      [ -n "$LATEST10" ] && ENVS="$ENVS GAZELLE_CALIB_10=/workspace/out/calib/$(basename "$LATEST10")"
      [ -n "$SCALAR9" ] && ENVS="$ENVS GAZELLE_BOARD_CALIB_9=$(basename "$SCALAR9")"
      [ -n "$SCALAR10" ] && ENVS="$ENVS GAZELLE_BOARD_CALIB_10=$(basename "$SCALAR10")"
      [ "$B" = "2" ] && ENVS="GAZELLE_FAKE=1 $ENVS"
      say "启动前端 (后台) → http://localhost:$FRONT_PORT"
      docker exec -d "$CONTAINER" bash -c \
        "cd /workspace/app && $ENVS nohup uvicorn demo.server.app:app --host 0.0.0.0 --port 8000 > /workspace/out/frontend.log 2>&1 &"
      sleep 2
      docker exec "$CONTAINER" bash -c "tail -3 /workspace/out/frontend.log 2>/dev/null"
      echo "  浏览器打开: http://localhost:$FRONT_PORT"
      echo "  (停止前端: docker exec $CONTAINER pkill -f uvicorn)"
      ;;
    3)
      # 生成 test200 npy (若不存在)
      if [ ! -f "$OUT_DIR/test200_images.npy" ]; then
        say "生成 test 200 张 npy (eurosat_split seed42)..."
        docker exec "$CONTAINER" python /workspace/app/tools/make_test200.py \
          --data-dir /workspace/eurosat_data --out /workspace/out/test200 --limit 200
      fi
      echo "  验证模型: [1] M9   [2] M10   [3] M9+M10"
      read -rp "  选择 [1-3]: " V
      echo "  后端: [1] 真机 http   [2] 离线 numpy 参考"
      read -rp "  选择 [1-2]: " VB
      BACKEND=http; [ "$VB" = "2" ] && BACKEND=numpy
      # 选择逐列校准 (若有)
      CAL=""
      CLS=$(ls "$OUT_DIR/calib"/calib_col_m*_demo_*.json 2>/dev/null)
      if [ -n "$CLS" ]; then
        echo "  可用逐列校准 (同窗口):"
        echo "$CLS" | nl
        read -rp "  输入编号 (0=不使用): " CN
        [ "${CN:-0}" != "0" ] && CAL=$(echo "$CLS" | sed -n "${CN}p")
      fi
      for MD in 9 10; do
        [ "$V" != "1" ] && [ "$MD" = "9" ] && continue
        [ "$V" != "2" ] && [ "$MD" = "10" ] && continue
        say "M$MD 抽样验证 (n=200, backend=$BACKEND)..."
        docker exec "$CONTAINER" python /workspace/app/tools/run_sample_verify.py \
          --model "model$MD" --backend "$BACKEND" --limit 200 \
          --images /workspace/out/test200_images.npy \
          --labels /workspace/out/test200_labels.npy \
          --calib-col "${CAL:-}" 2>&1 | grep -vE "^\s*$"
      done
      echo "  结果已保存: $OUT_DIR/logits_*.npy / errors_*.csv"
      ;;
    4) docker exec -it "$CONTAINER" bash /workspace/app/tools/board_connect.sh ;;
    5) echo "再见"; exit 0;;
    *) echo "  无效选择";;
  esac
done
