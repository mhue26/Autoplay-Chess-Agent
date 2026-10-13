"""SQLite ledger for RL-Chess: tracking checkpoints, games, steps, and Elo."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    generation INTEGER NOT NULL,
    parent_id INTEGER,
    path TEXT NOT NULL,
    sha256 TEXT,
    promoted INTEGER NOT NULL DEFAULT 0,
    config_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parent_id) REFERENCES checkpoints(id)
);

CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL, -- 'selfplay', 'arena', 'benchmark', 'human'
    white TEXT NOT NULL,
    black TEXT NOT NULL,
    result TEXT NOT NULL, -- '1-0', '0-1', '1/2-1/2'
    plies INTEGER NOT NULL,
    pgn_path TEXT,
    checkpoint_id INTEGER,
    metadata_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (checkpoint_id) REFERENCES checkpoints(id)
);

CREATE TABLE IF NOT EXISTS train_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    checkpoint_id INTEGER,
    generation INTEGER NOT NULL,
    step INTEGER NOT NULL,
    loss_total REAL NOT NULL,
    loss_policy REAL NOT NULL,
    loss_value REAL NOT NULL,
    lr REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (checkpoint_id) REFERENCES checkpoints(id)
);

CREATE TABLE IF NOT EXISTS elo_estimates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    checkpoint_id INTEGER NOT NULL,
    method TEXT NOT NULL, -- 'stockfish_anchor', 'relative_arena', 'puzzle_acc'
    elo REAL NOT NULL,
    ci_low REAL,
    ci_high REAL,
    n_games INTEGER NOT NULL,
    opponent_set TEXT,
    budget_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (checkpoint_id) REFERENCES checkpoints(id)
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    payload_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_games_kind ON games(kind);
CREATE INDEX IF NOT EXISTS idx_games_checkpoint ON games(checkpoint_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_gen ON checkpoints(generation);
CREATE INDEX IF NOT EXISTS idx_elo_checkpoint ON elo_estimates(checkpoint_id);
"""


@dataclass
class CheckpointRecord:
    id: int
    name: str
    generation: int
    parent_id: Optional[int]
    path: str
    sha256: Optional[str]
    promoted: bool
    config_json: Optional[str]
    created_at: str


class Ledger:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(SCHEMA)

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
        config_json = json.dumps(config) if config is not None else None
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO checkpoints (name, generation, parent_id, path, sha256, promoted, config_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (name, generation, parent_id, str(path), sha256, int(promoted), config_json),
            )
            checkpoint_id = cursor.lastrowid
            conn.execute(
                "INSERT INTO events (event_type, payload_json) VALUES (?, ?)",
                ("checkpoint_registered", json.dumps({"checkpoint_id": checkpoint_id, "name": name, "generation": generation})),
            )
            return checkpoint_id

    def promote_checkpoint(self, checkpoint_id: int) -> None:
        with self._get_connection() as conn:
            conn.execute("UPDATE checkpoints SET promoted = 1 WHERE id = ?", (checkpoint_id,))
            conn.execute(
                "INSERT INTO events (event_type, payload_json) VALUES (?, ?)",
                ("checkpoint_promoted", json.dumps({"checkpoint_id": checkpoint_id})),
            )

    def get_champion_checkpoint(self) -> Optional[CheckpointRecord]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM checkpoints WHERE promoted = 1 ORDER BY generation DESC, id DESC LIMIT 1"
            ).fetchone()
            if row is None:
                # Fallback to the latest registered checkpoint if none marked promoted
                row = conn.execute(
                    "SELECT * FROM checkpoints ORDER BY generation DESC, id DESC LIMIT 1"
                ).fetchone()
            if row is None:
                return None
            return CheckpointRecord(
                id=row["id"],
                name=row["name"],
                generation=row["generation"],
                parent_id=row["parent_id"],
                path=row["path"],
                sha256=row["sha256"],
                promoted=bool(row["promoted"]),
                config_json=row["config_json"],
                created_at=row["created_at"],
            )

    def list_checkpoints(self) -> List[CheckpointRecord]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM checkpoints ORDER BY generation ASC, id ASC").fetchall()
            return [
                CheckpointRecord(
                    id=r["id"],
                    name=r["name"],
                    generation=r["generation"],
                    parent_id=r["parent_id"],
                    path=r["path"],
                    sha256=r["sha256"],
                    promoted=bool(r["promoted"]),
                    config_json=r["config_json"],
                    created_at=r["created_at"],
                )
                for r in rows
            ]

    def record_game(
        self,
        kind: str,
        white: str,
        black: str,
        result: str,
        plies: int,
        pgn_path: Optional[str] = None,
        checkpoint_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        meta_json = json.dumps(metadata) if metadata is not None else None
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO games (kind, white, black, result, plies, pgn_path, checkpoint_id, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (kind, white, black, result, plies, pgn_path, checkpoint_id, meta_json),
            )
            return cursor.lastrowid

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
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO train_steps (checkpoint_id, generation, step, loss_total, loss_policy, loss_value, lr)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (checkpoint_id, generation, step, loss_total, loss_policy, loss_value, lr),
            )
            return cursor.lastrowid

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
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO elo_estimates (checkpoint_id, method, elo, ci_low, ci_high, n_games, opponent_set, budget_notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (checkpoint_id, method, elo, ci_low, ci_high, n_games, opponent_set, budget_notes),
            )
            return cursor.lastrowid

    def get_stats_summary(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            total_games = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
            selfplay_games = conn.execute("SELECT COUNT(*) FROM games WHERE kind = 'selfplay'").fetchone()[0]
            arena_games = conn.execute("SELECT COUNT(*) FROM games WHERE kind = 'arena'").fetchone()[0]
            benchmark_games = conn.execute("SELECT COUNT(*) FROM games WHERE kind = 'benchmark'").fetchone()[0]
            human_games = conn.execute("SELECT COUNT(*) FROM games WHERE kind = 'human'").fetchone()[0]

            total_plies = conn.execute("SELECT COALESCE(SUM(plies), 0) FROM games WHERE kind = 'selfplay'").fetchone()[0]
            train_steps_count = conn.execute("SELECT COUNT(*) FROM train_steps").fetchone()[0]
            checkpoints_count = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]

            latest_champ = self.get_champion_checkpoint()
            latest_elo = conn.execute(
                """
                SELECT e.*, c.name as checkpoint_name
                FROM elo_estimates e
                JOIN checkpoints c ON e.checkpoint_id = c.id
                ORDER BY e.id DESC LIMIT 1
                """
            ).fetchone()

            return {
                "total_games": total_games,
                "selfplay_games": selfplay_games,
                "selfplay_positions_estimated": total_plies,
                "arena_games": arena_games,
                "benchmark_games": benchmark_games,
                "human_games": human_games,
                "training_steps": train_steps_count,
                "checkpoints_count": checkpoints_count,
                "champion": {
                    "name": latest_champ.name if latest_champ else None,
                    "generation": latest_champ.generation if latest_champ else None,
                    "path": latest_champ.path if latest_champ else None,
                } if latest_champ else None,
                "latest_elo": dict(latest_elo) if latest_elo else None,
            }

    def get_elo_history(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT e.id, e.method, e.elo, e.ci_low, e.ci_high, e.n_games,
                       e.opponent_set, e.budget_notes, e.created_at,
                       c.name as checkpoint_name, c.generation
                FROM elo_estimates e
                JOIN checkpoints c ON e.checkpoint_id = c.id
                ORDER BY e.created_at ASC, e.id ASC
                """
            ).fetchall()
            return [dict(r) for r in rows]
