"""sys.path bootstrap for demo tests.

Repo convention: entry scripts put ``src/`` on sys.path then ``import _pathsetup``
to add ``src/{core,qat,data,training,scripts}`` (flat imports like
``from optic_layers import ...``).  Tests additionally need ``demo/remote`` and
the repo root (for ``demo.server.app`` package imports).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))          # demo/tests
DEMO_DIR = os.path.dirname(_HERE)                           # demo/
REPO_ROOT = os.path.dirname(DEMO_DIR)                       # repo root

for _p in (
    REPO_ROOT,
    os.path.join(REPO_ROOT, "src"),
    os.path.join(DEMO_DIR, "remote"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _pathsetup  # noqa: E402,F401  (adds src/core, src/data, ...)

# Test collection must happen from the repo root so that relative data/weight
# paths resolve; guard early with a clear error instead of cryptic failures.
assert os.path.isdir(os.path.join(REPO_ROOT, "data", "EuroSAT_RGB")), \
    "run tests from the repo root (data/EuroSAT_RGB missing)"
