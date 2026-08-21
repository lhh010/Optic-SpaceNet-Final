#!/bin/bash
# Run on Gazelle board as root: sudo bash check_env.sh
echo "=== process check ==="
ps aux | grep -c '[c]ompass'
echo "=== numpy/scipy ==="
python3 -c 'import numpy, scipy; print("numpy=", numpy.__version__, "scipy=", scipy.__version__)'
python3 -c 'from scipy._lib import decorator; print("scipy._lib.decorator OK")' 2>&1 | tail -1
echo "=== net check ==="
timeout 5 bash -c 'cat < /dev/null > /dev/tcp/archive.ubuntu.com/80' 2>/dev/null && echo NET_OK || echo NET_FAIL
timeout 5 bash -c 'cat < /dev/null > /dev/tcp/pypi.org/443' 2>/dev/null && echo PYPI_OK || echo PYPI_FAIL
echo "=== apt cache ==="
ls /var/cache/apt/archives/*.deb 2>/dev/null | head -5
echo "=== scipy files ==="
ls /usr/lib/python3/dist-packages/scipy/_lib/ | head -20
echo "=== dpkg scipy ==="
dpkg -l | grep -E "scipy|numpy"
