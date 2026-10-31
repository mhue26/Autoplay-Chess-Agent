"""Tests for SQLite ledger operations."""

from __future__ import annotations

import tempfile
from pathlib import Path

from rl_chess.ledger import Ledger


def test_ledger_flow():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_ledger.db"
        ledger = Ledger(db_path)

        # 1. Record initial checkpoint
        c1_id = ledger.record_checkpoint(
            name="chk_gen0",
            generation=0,
            path="data/checkpoints/gen0.pt",
            promoted=True,
            config={"network": {"num_blocks": 10, "channels": 128}},
        )
        assert c1_id == 1

        champ = ledger.get_champion_checkpoint()
        assert champ is not None
        assert champ.name == "chk_gen0"
        assert champ.generation == 0

        # 2. Record self-play games
        g1 = ledger.record_game(
            kind="selfplay",
            white="chk_gen0",
            black="chk_gen0",
            result="1-0",
            plies=42,
            checkpoint_id=c1_id,
        )
        assert g1 == 1

        g2 = ledger.record_game(
            kind="human",
            white="human",
            black="chk_gen0",
            result="0-1",
            plies=30,
            checkpoint_id=c1_id,
        )
        assert g2 == 2

        # 3. Record training steps
        s1 = ledger.record_train_step(
            generation=1,
            step=10,
            loss_total=1.23,
            loss_policy=0.8,
            loss_value=0.43,
            lr=0.001,
            checkpoint_id=c1_id,
        )
        assert s1 == 1

        # 4. Record new checkpoint and promote
        c2_id = ledger.record_checkpoint(
            name="chk_gen1",
            generation=1,
            parent_id=c1_id,
            path="data/checkpoints/gen1.pt",
            promoted=False,
        )
        assert c2_id == 2
        # Champion is still gen 0 until promoted
        assert ledger.get_champion_checkpoint().id == c1_id

        ledger.promote_checkpoint(c2_id)
        assert ledger.get_champion_checkpoint().id == c2_id

        # 5. Record Elo estimate
        e1 = ledger.record_elo_estimate(
            checkpoint_id=c2_id,
            method="stockfish_anchor",
            elo=1150.0,
            n_games=20,
            ci_low=1080.0,
            ci_high=1220.0,
            opponent_set="Stockfish-Skill0,Skill5",
            budget_notes="sims=64",
        )
        assert e1 == 1

        # 6. Verify summary
        summary = ledger.get_stats_summary()
        assert summary["total_games"] == 2
        assert summary["selfplay_games"] == 1
        assert summary["selfplay_positions_estimated"] == 42
        assert summary["human_games"] == 1
        assert summary["training_steps"] == 1
        assert summary["checkpoints_count"] == 2
        assert summary["champion"]["name"] == "chk_gen1"

        elo_hist = ledger.get_elo_history()
        assert len(elo_hist) == 1
        assert elo_hist[0]["checkpoint_name"] == "chk_gen1"
        assert elo_hist[0]["elo"] == 1150.0
