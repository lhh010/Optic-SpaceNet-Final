#!/bin/bash
# Inspect scipy._lib.decorator import failure
echo "=== scipy._lib __init__ ==="
cat /usr/lib/python3/dist-packages/scipy/_lib/__init__.py
echo "=== decorator.py (first 40 lines) ==="
head -40 /usr/lib/python3/dist-packages/scipy/_lib/decorator.py
echo "=== import attempts ==="
python3 -c 'import scipy._lib; print("scipy._lib path:", scipy._lib.__path__)' 2>&1 | tail -2
python3 -c 'from scipy._lib import decorator' 2>&1 | tail -3
python3 -c 'import importlib; m = importlib.import_module("scipy._lib.decorator"); print("importlib OK")' 2>&1 | tail -3
echo "=== pyc check ==="
ls -la /usr/lib/python3/dist-packages/scipy/_lib/__pycache__/ 2>/dev/null | grep -i decor
echo "=== other decorator copies ==="
find /usr -name "decorator.py" -path "*scipy*" 2>/dev/null
