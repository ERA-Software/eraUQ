"""
Pytest configuration for legacy end-to-end tests.
Ensures eraUQ package is importable when running tests from project root.
"""
import json
import sys
from pathlib import Path

import numpy as np

# Add project src to path so "from eraUQ import ..." works when running pytest
# from project root (e.g. pytest legacy_examples/end_to_end_tests/)
_root = Path(__file__).resolve().parents[2]
_src = _root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


def load_ref_json(path):
    """Load a reference JSON file; return dict with arrays as numpy arrays, scalars as float."""
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        if isinstance(v, list):
            out[k] = np.array(v)
        else:
            out[k] = v
    return out
