"""YAML config loading and default paths."""

from __future__ import annotations

from typing import Any, Dict

import yaml

DEFAULT_CONFIG_PATH = "configs/config.yaml"

_PATH_DEFAULTS = {
    "ledger_db": "data/ledger.db",
    "checkpoints_dir": "data/checkpoints",
    "replay_dir": "data/replay",
    "pgns_dir": "data/pgns",
}


def load_config(path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_paths(cfg: Dict[str, Any]) -> Dict[str, str]:
    return {**_PATH_DEFAULTS, **cfg.get("paths", {})}
