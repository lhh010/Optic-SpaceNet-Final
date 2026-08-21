#!/bin/bash
# Fix + verify: install python3-decorator if missing, then test scipy.signal
echo "=== decorator check ==="
python3 -c 'import decorator; print("decorator", decorator.__version__)' 2>&1 | tail -1
dpkg -l | grep -i decorator || echo "python3-decorator NOT installed"
echo "=== install if missing ==="
if ! python3 -c 'import decorator' 2>/dev/null; then
  apt-get install -y python3-decorator 2>&1 | tail -3
fi
echo "=== retest scipy.signal ==="
python3 -c 'import scipy.signal; from scipy.signal import savgol_filter; print("scipy.signal OK")' 2>&1 | tail -1
echo "=== retest full scipy import path used by cali ==="
python3 -c 'import scipy; print("scipy", scipy.__version__, "full import OK")' 2>&1 | tail -1
