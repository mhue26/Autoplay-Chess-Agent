"""Shared pytest fixtures (isolated data dirs so GUI tests never touch data/ledger.db)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rl_chess.config import DEFAULT_CONFIG_PATH, load_config


@pytest.fixture
def app_config_path(tmp_path: Path) -> str:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    root = tmp_path / "data"
    paths = {
        "ledger_db": str(root / "ledger.db"),
        "checkpoints_dir": str(root / "checkpoints"),
        "replay_dir": str(root / "replay"),
        "pgns_dir": str(root / "pgns"),
    }
    for key, value in paths.items():
        p = Path(value)
        if key != "ledger_db":
            p.mkdir(parents=True, exist_ok=True)

    cfg["paths"] = paths
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return str(config_file)
