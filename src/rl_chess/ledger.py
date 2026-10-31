"""SQLite ledger: checkpoints, games, training steps, and Elo estimates."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    generation INTEGER NOT NULL,
    parent_id INTEGER REFERENCES checkpoints(id),
    path TEXT NOT NULL,
    sha256 TEXT,
    promoted INTEGER NOT NULL DEFAULT 0,
    config_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL, -- 'selfplay', 'arena', 'benchmark', 'human'
    white TEXT NOT NULL,
    black TEXT NOT NULL,
    result TEXT NOT NULL, -- '1-0', '0-1', '1/2-1/2'
    plies INTEGER NOT NULL,
    pgn_path TEXT,
    checkpoint_id INTEGER REFERENCES checkpoints(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS train_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    checkpoint_id INTEGER REFERENCES checkpoints(id),
    generation INTEGER NOT NULL,
    step INTEGER NOT NULL,
    loss_total REAL NOT NULL,
    loss_policy REAL NOT NULL,
    loss_value REAL NOT NULL,
    lr REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS elo_estimates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    checkpoint_id INTEGER NOT NULL REFERENCES checkpoints(id),
    method TEXT NOT NULL,
    elo REAL NOT NULL,
    ci_low REAL,
    ci_high REAL,
    n_games INTEGER NOT NULL,
    opponent_set TEXT,
    budget_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_games_kind ON games(kind);
CREATE INDEX IF NOT EXISTS idx_elo_checkpoint ON elo_estimates(checkpoint_id);
"""


@dataclass
class CheckpointRecord:
    id: int
    name: str
    generation: int
    path: str
    promoted: bool


class Ledger:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            with conn:  # commit or roll back
                yield conn
        finally:
            conn.close()

    def _insert(self, sql: str, params: tuple) -> int:
        with self._connect() as conn:
            return conn.execute(sql, params).lastrowid

    @staticmethod
    def _checkpoint(row: sqlite3.Row) -> CheckpointRecord:
        return CheckpointRecord(row["id"], row["name"], row["generation"], row["path"], bool(row["promoted"]))

    # -- checkpoints -----------------------------------------------------

    def record_checkpoint(
        self,
        name: str,
        generation: int,
        path: str,
        parent_id: Optional[int] = None,
        sha256: Optional[str] = None,
        promoted: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ) -> int:
        return self._insert(
            "INSERT INTO checkpoints (name, generation, parent_id, path, sha256, promoted, config_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (name, generation, parent_id, str(path), sha256, int(promoted),
             json.dumps(config) if config is not None else None),
        )

    def promote_checkpoint(self, checkpoint_id: int) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE checkpoints SET promoted = 1 WHERE id = ?", (checkpoint_id,))

    def wipe_training_history(self) -> None:
        """Remove checkpoints, games, training steps, and Elo rows (fresh training run)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM elo_estimates")
            conn.execute("DELETE FROM train_steps")
            conn.execute("DELETE FROM games")
            conn.execute("DELETE FROM checkpoints")

    def get_champion_checkpoint(self) -> Optional[CheckpointRecord]:
        """Latest promoted checkpoint, falling back to the latest registered one."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM checkpoints ORDER BY promoted DESC, generation DESC, id DESC LIMIT 1"
            ).fetchone()
        return self._checkpoint(row) if row else None

    def list_checkpoints(self) -> List[CheckpointRecord]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM checkpoints ORDER BY generation, id").fetchall()
        return [self._checkpoint(r) for r in rows]

    def resolve_checkpoint(self, target: str = "") -> Optional[CheckpointRecord]:
        """Find a checkpoint by registered name or file path; empty means the champion."""
        if not target:
            return self.get_champion_checkpoint()
        return next(
            (c for c in self.list_checkpoints() if target in (c.name, c.path) or Path(target) == Path(c.path)),
            None,
        )

    # -- records ---------------------------------------------------------

    def record_game(
        self,
        kind: str,
        white: str,
        black: str,
        result: str,
        plies: int,
        pgn_path: Optional[str] = None,
        checkpoint_id: Optional[int] = None,
    ) -> int:
        return self._insert(
            "INSERT INTO games (kind, white, black, result, plies, pgn_path, checkpoint_id)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (kind, white, black, result, plies, pgn_path, checkpoint_id),
        )

    def record_train_step(
        self,
        generation: int,
        step: int,
        loss_total: float,
        loss_policy: float,
        loss_value: float,
        lr: float,
        checkpoint_id: Optional[int] = None,
    ) -> int:
        return self._insert(
            "INSERT INTO train_steps (checkpoint_id, generation, step, loss_total, loss_policy, loss_value, lr)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (checkpoint_id, generation, step, loss_total, loss_policy, loss_value, lr),
        )

    def record_elo_estimate(
        self,
        checkpoint_id: int,
        method: str,
        elo: float,
        n_games: int,
        ci_low: Optional[float] = None,
        ci_high: Optional[float] = None,
        opponent_set: Optional[str] = None,
        budget_notes: Optional[str] = None,
    ) -> int:
        return self._insert(
            "INSERT INTO elo_estimates (checkpoint_id, method, elo, ci_low, ci_high, n_games, opponent_set, budget_notes)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (checkpoint_id, method, elo, ci_low, ci_high, n_games, opponent_set, budget_notes),
        )

    # -- reporting -------------------------------------------------------

    def get_stats_summary(self) -> Dict[str, Any]:
        with self._connect() as conn:
            games = {r["kind"]: r["n"] for r in conn.execute("SELECT kind, COUNT(*) AS n FROM games GROUP BY kind")}
            selfplay_plies = conn.execute(
                "SELECT COALESCE(SUM(plies), 0) FROM games WHERE kind = 'selfplay'"
            ).fetchone()[0]
            train_steps = conn.execute("SELECT COUNT(*) FROM train_steps").fetchone()[0]
            checkpoints = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
            latest_elo = conn.execute(
                "SELECT e.*, c.name AS checkpoint_name FROM elo_estimates e"
                " JOIN checkpoints c ON e.checkpoint_id = c.id ORDER BY e.id DESC LIMIT 1"
            ).fetchone()

        champ = self.get_champion_checkpoint()
        return {
            "total_games": sum(games.values()),
            "selfplay_games": games.get("selfplay", 0),
            "selfplay_positions_estimated": selfplay_plies,
            "arena_games": games.get("arena", 0),
            "benchmark_games": games.get("benchmark", 0),
            "human_games": games.get("human", 0),
            "training_steps": train_steps,
            "checkpoints_count": checkpoints,
            "champion": {"name": champ.name, "generation": champ.generation, "path": champ.path} if champ else None,
            "latest_elo": dict(latest_elo) if latest_elo else None,
        }

    def get_elo_history(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT e.id, e.method, e.elo, e.ci_low, e.ci_high, e.n_games, e.opponent_set,"
                " e.budget_notes, e.created_at, c.name AS checkpoint_name, c.generation"
                " FROM elo_estimates e JOIN checkpoints c ON e.checkpoint_id = c.id"
                " ORDER BY e.created_at, e.id"
            ).fetchall()
        return [dict(r) for r in rows]
