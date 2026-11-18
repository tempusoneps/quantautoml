"""Load a bundle's standalone ``predictor.py`` without leaving bytecode inside the bundle."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from quantautoml.common.exceptions import PackagingError


def load_predictor(path: Path) -> ModuleType:
    """Import ``path`` (a bundle's predictor.py) as a throw-away module."""
    spec = importlib.util.spec_from_file_location(f"_bundle_predictor_{abs(hash(str(path)))}", path)
    if spec is None or spec.loader is None:
        raise PackagingError(f"cannot load predictor at {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # never leave __pycache__ inside the bundle
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module
