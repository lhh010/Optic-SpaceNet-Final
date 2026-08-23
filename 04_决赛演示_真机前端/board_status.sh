#!/usr/bin/env bash
# 板上部署/前端情况一键检查 (在板子上执行: bash board_status.sh)
echo '==== 1. 当前用户/主机 ===='
whoami; hostname; uname -a

echo; echo '==== 2. 他队进程检测 (gazelle/compass/server) ===='
ps aux | grep -iE 'gazelle|compass|server_gazelle|run_j1|run_ds3' | grep -v grep || echo '(无相关进程)'

echo; echo '==== 3. 端口监听 (22 / 8000) ===='
ss -tlnp 2>/dev/null | grep -E ':22|:8000' || netstat -tlnp 2>/dev/null | grep -E ':22|:8000'

echo; echo '==== 4. server_gazelle 是否在 8000 (curl 探活) ===='
if command -v curl >/dev/null 2>&1; then
  curl -s -m 3 http://127.0.0.1:8000/health && echo || echo '(8000 未响应: server_gazelle 未起)'
else
  echo '(无 curl; 用 python)'; python3 -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3).read())" 2>&1 | head -c 300
fi

echo; echo '==== 5. 部署目录 (/home/uisrc) 结构 ===='
ls -la ~/ 2>/dev/null
echo '--- 找 server_gazelle / run_ds3 / export / weights ---'
find ~ /opt /home -maxdepth 3 -iname '*server_gazelle*' -o -iname 'run_ds3*' -o -iname 'run_j1*' -o -iname 'export_ds3*' 2>/dev/null | head -40

echo; echo '==== 6. 权重/校准/测试数据 (j1/weights_ds3pool3 等) ===='
find ~ -maxdepth 3 \( -iname '*weights_ds3*' -o -iname '*weights_c3d*' -o -iname '*calib*json' -o -iname '*test_images*' -o -iname 'meta.json' \) 2>/dev/null | head -40

echo; echo '==== 7. 当前 server_gazelle 前台是否已运行(本窗口) ===='
pgrep -af server_gazelle || echo '(未运行)'

echo; echo '==== 8. 提示 ===='
echo '若 8000 未起:  cd 到含 server_gazelle.py 的目录后执行'
echo '  sudo env OPTC_PORT=8000 python3 server_gazelle.py'