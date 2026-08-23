#!/bin/bash
# ==============================================================================
# Gazelle 光计算硬件连接与状态检查 (容器内运行; 真机直连, 不用 osimulator)
# 连接方式:
#   1) SSH   : ssh uisrc@192.168.31.158 (密码 5182); 板上 server_gazelle.py :8000
#   2) 串口  : 本机 USB 转串口 (/dev/ttyUSB*) 直连板 console, 115200 8N1
#              picocom -b 115200 /dev/ttyUSB0
#              (账号 uisrc / 密码 root, 见产品手册 3.1.1; ifconfig 查 IP 后转 SSH)
# 功能: 连通性检查 / 板上服务状态 / 启动-停止 server_gazelle / EBR / 串口终端
# ==============================================================================
set -u
BOARD_HOST="${GAZELLE_HOST:-192.168.31.158}"
BOARD_USER="uisrc"
BOARD_PASS="5182"

sshx() { sshpass -p "$BOARD_PASS" ssh -o StrictHostKeyChecking=no \
          -o ConnectTimeout=15 "$BOARD_USER@$BOARD_HOST" "$@"; }

srv_status() {
  echo "---- 板上进程/端口 ----"
  sshx "who; echo ---; ps aux | grep -iE 'server_gazelle|compass' | grep -v grep | head -5; \
        echo ---; (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -E ':8000|:22' | head -4; \
        echo ---; ls -la /home/uisrc/api.log 2>/dev/null | awk '{print \$5, \$9}'"
}

srv_start() {
  echo "---- 启动板上 server_gazelle.py (root, :8000) ----"
  sshx "cd /home/uisrc/opticspacenet 2>/dev/null && \
        ls server_gazelle.py 2>/dev/null && \
        echo $BOARD_PASS | sudo -S -p X nohup python3 server_gazelle.py > /tmp/srv_demo.log 2>&1 & \
        sleep 3; \
        ps aux | grep server_gazelle | grep -v grep | head -2; \
        tail -2 /tmp/srv_demo.log 2>/dev/null"
}

srv_stop() {
  echo "---- 停止板上 server_gazelle.py ----"
  sshx "echo $BOARD_PASS | sudo -S -p X pkill -f server_gazelle; sleep 1; \
        ps aux | grep server_gazelle | grep -v grep | wc -l"
}

ebr_check() {
  echo "---- EBR 快速检查 (compass_evb_test) ----"
  sshx "cd /home/uisrc/sample_code/code 2>/dev/null && \
        echo $BOARD_PASS | sudo -S -p X timeout 600 python3 evb_test_sample.py 2>/dev/null | \
        tr '\r' '\n' | grep -aE 'error_std|ebr' | tail -2"
}

serial_term() {
  echo "---- 本机串口设备 ----"
  DEVS=$(ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null)
  if [ -z "$DEVS" ]; then
    echo "!! 未发现 /dev/ttyUSB* 或 /dev/ttyACM* (检查 USB 转串口线/CP210x 驱动)"
    echo "   容器启动时挂载的串口设备才会显示; 若设备后插, 需重建容器"
    return 1
  fi
  echo "$DEVS"
  ls -l /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
  echo "  (若 dialout 组无权限: 宿主机 usermod -aG dialout \$USER 后重进会话)"
  read -rp "选择串口 (如 /dev/ttyUSB0, 默认第一个): " D
  [ -n "$D" ] || D=$(echo "$DEVS" | head -1)
  echo "---- 打开 $D @ 115200 (picocom; 退出 Ctrl+A Ctrl+X) ----"
  echo "   板 console 账号 uisrc / 密码 root (产品手册 3.1.1)"
  echo "   登录后 ifconfig 查板 IP → 转 SSH"
  read -rp "回车开始..."
  picocom -b 115200 -d 8 -p n -s 1 "$D" 2>&1
  rc=$?
  if [ $rc -ne 0 ] && command -v python3 >/dev/null; then
    echo "!! picocom 退出 ($rc), 尝试 pyserial 简易终端..."
    python3 - "$D" <<'PYEOF'
import sys
import serial

dev = sys.argv[1]
try:
    ser = serial.Serial(dev, 115200, timeout=0.5)
except Exception as e:
    print("!! 打开失败: %s (权限? 设备被占用?)" % e)
    sys.exit(1)
print("已打开 %s @115200 (输入命令发送; Ctrl+C 退出)" % dev)
try:
    while True:
        line = input("> ")
        ser.write((line + "\r\n").encode())
        out = ser.read(4096).decode(errors="replace")
        if out:
            print(out, end="")
except (KeyboardInterrupt, EOFError):
    pass
ser.close()
print("\n串口已关闭")
PYEOF
  fi
}

while true; do
  echo ""
  echo "================ Gazelle 硬件连接与状态 ================"
  echo "  连接目标: SSH uisrc@$BOARD_HOST (:8000 server_gazelle)"
  echo "  [1] SSH 连通性 + 板上服务状态检查"
  echo "  [2] 启动板上 server_gazelle.py (root)"
  echo "  [3] 停止板上 server_gazelle.py"
  echo "  [4] EBR 快速检查 (evb_test)"
  echo "  [5] 串口直连 (picocom -b 115200 /dev/ttyUSB0 → 板 console)"
  echo "  [6] 返回主菜单"
  echo "======================================================"
  read -rp "请选择 [1-6]: " C
  case "$C" in
    1) srv_status ;;
    2) srv_start ;;
    3) srv_stop ;;
    4) ebr_check ;;
    5) serial_term ;;
    6) exit 0 ;;
    *) echo "  无效选择";;
  esac
done
