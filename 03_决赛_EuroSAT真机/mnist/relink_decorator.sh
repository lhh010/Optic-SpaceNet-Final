#!/bin/bash
# Recreate scipy._lib.decorator symlink pointing at the working decorator module
echo "=== locate decorator ==="
python3 -c 'import decorator; print(decorator.__file__); print(decorator.__version__)' 2>&1 | tail -2
DEC=$(python3 -c 'import decorator; print(decorator.__file__)' 2>/dev/null)
echo "DEC=$DEC"
echo "=== recreate symlink ==="
ln -sf "$DEC" /usr/lib/python3/dist-packages/scipy/_lib/decorator.py
ls -la /usr/lib/python3/dist-packages/scipy/_lib/decorator.py
echo "=== retest ==="
python3 -c 'from scipy._lib.decorator import decorator; print("decorator import OK")' 2>&1 | tail -2
python3 -c 'import scipy.signal; from scipy.signal import savgol_filter; print("scipy.signal OK")' 2>&1 | tail -2
