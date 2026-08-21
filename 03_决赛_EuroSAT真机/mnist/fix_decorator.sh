#!/bin/bash
# Fix broken scipy._lib.decorator symlink (source missing, pyc intact)
echo "=== before ==="
ls -la /usr/lib/python3/dist-packages/scipy/_lib/decorator.py 2>&1
echo "=== remove broken symlink/source ==="
rm -f /usr/lib/python3/dist-packages/scipy/_lib/decorator.py
echo "=== retest ==="
python3 -c 'from scipy._lib.decorator import decorator; print("decorator import OK")' 2>&1 | tail -2
python3 -c 'import scipy.signal; from scipy.signal import savgol_filter; print("scipy.signal OK")' 2>&1 | tail -2
python3 -c 'import scipy; print("scipy", scipy.__version__)' 2>&1 | tail -1
