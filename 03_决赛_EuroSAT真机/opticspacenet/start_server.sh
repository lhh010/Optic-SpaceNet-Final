#!/bin/bash
# Start (or restart) the optical matmul server on the Gazelle board.
# Run as root:  sudo bash start_server.sh
cd /home/uisrc/opticspacenet
pkill -f "[s]erver_gazelle.py"
sleep 1
setsid nohup python3 server_gazelle.py > server.log 2>&1 < /dev/null &
sleep 25
echo "--- server.log tail ---"
tail -4 server.log
echo "--- process ---"
ps aux | grep "[s]erver_gazelle" | head -2
echo "--- port ---"
netstat -tlnp 2>/dev/null | grep 8000 || echo "(netstat unavailable)"
