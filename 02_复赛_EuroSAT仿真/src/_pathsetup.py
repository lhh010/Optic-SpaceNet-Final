"""sys.path bootstrap for the src/ layout.

After the project restructure, the codebase still uses flat imports
(``from optic_qat_v4 import ...``, ``from optic_layers import ...``,
``import osimulator``). For those to resolve, every ``src/<subdir>`` and the
repo root must be on ``sys.path``.

Each entry-point script does this with a small prologue that first puts ``src/``
on ``sys.path`` (so this module is importable) and then ``import _pathsetup``
to add the rest:

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import _pathsetup  # noqa: E402,F401

Importable library modules (``optic_layers``, ``optic_qat*``, ``eurosat_split``)
do not need the prologue — they are always imported after an entry script has
already run it. This module is idempotent.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))   # .../src
_ROOT = os.path.dirname(_HERE)                        # repo root

for _sub in ("core", "qat", "data", "training", "scripts"):
    _p = os.path.join(_HERE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)   # keeps `import osimulator` (+ CWD-relative data/weights) resolving
