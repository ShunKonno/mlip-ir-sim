"""Make the ``src`` layout importable when running the test suite directly.

Ensures ``import mlip_ir_sim`` works with a plain ``pytest`` invocation even
without an editable install (``pip install -e .``).
"""
import sys
from pathlib import Path

_SRC = Path(__file__).parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
