"""Reset training state: replay buffer, ledger, and on-disk checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from rl_chess.ledger import Ledger


def reset_agent_to_gen_zero(cfg: Dict[str, Any], paths: Dict[str, str], ledger: Ledger) -> None:
    """Clear replay memory, delete all checkpoint files, and wipe the training ledger.

    Does not register a new checkpoint; the next training run will create gen_0 if needed.
    """
    checkpoints_dir = Path(paths["checkpoints_dir"])
    replay_dir = Path(paths["replay_dir"])
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    replay_dir.mkdir(parents=True, exist_ok=True)

    buffer_file = replay_dir / "replay_buffer.npz"
    if buffer_file.exists():
        buffer_file.unlink()

    for pt_file in checkpoints_dir.glob("*.pt"):
        pt_file.unlink()

    ledger.wipe_training_history()
