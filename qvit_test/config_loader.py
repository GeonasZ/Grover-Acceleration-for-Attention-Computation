"""
Load and expose settings from config.json.
All attention-filter and grover/topk options are read from here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

# Default path: project root / config.json (parent of qvit_test)
_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG_PATH = _THIS_DIR.parent / "config.json"

_CACHE: Dict[str, Any] | None = None


def _default_config() -> Dict[str, Any]:
    return {
        "training_option": {
            "attention_filter": "grover",
        },
        "grover": {
            "threshold": 0.0482,
            "use_qiskit": True,
            "grover_backend": "shortcut",
            "max_qubits": 5,
            "shots": None,
            "enable_filter": True,
            "percentile": 0.9,
            "neighbor_radius": 1,
        },
        "topk": {
            "k": 8,
            "enable_filter": True,
            "neighbor_radius": 1,
        },
    }


def get_config_path() -> Path:
    return Path(os.environ.get("QVIT_CONFIG_PATH", str(_DEFAULT_CONFIG_PATH)))


def load_config(path: Path | None = None) -> Dict[str, Any]:
    """Load config from JSON, merge with defaults. Cached after first load."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    path = path or get_config_path()
    defaults = _default_config()
    if not path.is_file():
        _CACHE = defaults
        return _CACHE
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        _CACHE = defaults
        return _CACHE
    # Deep merge: top-level keys from file override defaults
    def merge(base: Dict, override: Dict) -> Dict:
        out = dict(base)
        for k, v in override.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = merge(out[k], v)
            else:
                out[k] = v
        return out

    _CACHE = merge(defaults, data)
    return _CACHE


def get_training_option() -> Dict[str, Any]:
    return load_config().get("training_option", _default_config()["training_option"])


def get_attention_filter() -> str:
    """One of 'grover', 'topk'."""
    return get_training_option().get("attention_filter", "grover")


def get_grover_config() -> Dict[str, Any]:
    return load_config().get("grover", _default_config()["grover"])


def get_topk_config() -> Dict[str, Any]:
    return load_config().get("topk", _default_config()["topk"])


def reset_config_cache() -> None:
    """Clear cached config (e.g. for tests)."""
    global _CACHE
    _CACHE = None
