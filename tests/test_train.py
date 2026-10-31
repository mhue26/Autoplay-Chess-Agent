"""Tests for training components: replay buffer, self-play, trainer, arena."""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path
import numpy as np

from rl_chess.encoding import ACTION_SPACE_SIZE, NUM_INPUT_CHANNELS
from rl_chess.mcts import MCTS
from rl_chess.model import create_model
from rl_chess.train.games import evaluate_candidate, play_selfplay_game
from rl_chess.train.replay import ReplayBuffer
from rl_chess.train.reset import reset_agent_to_gen_zero
from rl_chess.train.loop import resolve_arena_games
from rl_chess.train.trainer import Trainer
from rl_chess.config import get_paths
from rl_chess.ledger import Ledger


def test_reset_agent_to_gen_zero():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        cfg = {
            "device": "cpu",
            "network": {"num_blocks": 2, "channels": 16},
            "paths": {
                "ledger_db": str(tmp / "ledger.db"),
                "checkpoints_dir": str(tmp / "ckpt"),
                "replay_dir": str(tmp / "replay"),
                "pgns_dir": str(tmp / "pgns"),
            },
        }
        paths = get_paths(cfg)
        ledger = Ledger(paths["ledger_db"])
        ckpt_dir = Path(paths["checkpoints_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        old_pt = ckpt_dir / "gen_3.pt"
        old_pt.write_bytes(b"fake")
        ledger.record_checkpoint("old", 3, str(old_pt), promoted=True)

        buf_path = Path(paths["replay_dir"]) / "replay_buffer.npz"
        buf_path.parent.mkdir(parents=True, exist_ok=True)
        ReplayBuffer(capacity=10).save(buf_path)

        reset_agent_to_gen_zero(cfg, paths, ledger)

        assert not buf_path.exists()
        assert not any(ckpt_dir.glob("*.pt"))
        assert len(ledger.list_checkpoints()) == 0
        assert ledger.get_champion_checkpoint() is None


def test_replay_buffer_and_trainer():
    with tempfile.TemporaryDirectory() as tmp_dir:
        buf = ReplayBuffer(capacity=100)
        # Push 10 fake samples
        for _ in range(10):
            st = np.random.randn(NUM_INPUT_CHANNELS, 8, 8).astype(np.float32)
            pi = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
            pi[np.random.randint(0, ACTION_SPACE_SIZE)] = 1.0
            val = float(np.random.choice([-1.0, 0.0, 1.0]))
            buf.push(st, pi, val)

        assert len(buf) == 10

        # Test save & load
        path = Path(tmp_dir) / "buf.npz"
        buf.save(path)
        buf2 = ReplayBuffer(capacity=100)
        buf2.load(path)
        assert len(buf2) == 10

        # Test trainer
        model = create_model(6, 64, device="cpu")
        trainer = Trainer(model, learning_rate=0.01, device="cpu")
        metrics = trainer.train_step(buf2, batch_size=4)
        assert metrics["loss_total"] > 0.0
        assert metrics["loss_policy"] > 0.0


def test_selfplay_short_game():
    with tempfile.TemporaryDirectory() as tmp_dir:
        pgn_file = Path(tmp_dir) / "test.pgn"
        net = create_model(6, 64, device="cpu")
        mcts = MCTS(net, device="cpu")

        samples, result, plies = play_selfplay_game(
            mcts=mcts,
            num_simulations=5,
            max_plies=4,
            pgn_path=pgn_file,
        )

        assert len(samples) == plies
        assert plies <= 4
        assert result in ["1-0", "0-1", "1/2-1/2"]
        assert pgn_file.exists()


def test_arena_short_match():
    net1 = create_model(6, 64, device="cpu")
    net2 = create_model(6, 64, device="cpu")

    promoted, win_rate, records = evaluate_candidate(
        candidate_net=net1,
        champion_net=net2,
        num_games=2,
        num_simulations=5,
        threshold=0.5,
        device="cpu",
    )

    assert len(records) == 2
    assert 0.0 <= win_rate <= 1.0
    assert isinstance(promoted, bool)


def test_resolve_arena_games_scales_with_self_play_cap():
    cfg = {"max_games": 20}
    assert resolve_arena_games(5, cfg) == 5
    assert resolve_arena_games(10, cfg) == 10
    assert resolve_arena_games(25, cfg) == 20
    assert resolve_arena_games(50, cfg) == 20
    assert resolve_arena_games(10, {"num_games": 15}) == 10


def test_selfplay_instant_stop():
    with tempfile.TemporaryDirectory() as tmp_dir:
        pgn_file = Path(tmp_dir) / "test.pgn"
        net = create_model(6, 64, device="cpu")
        mcts = MCTS(net, device="cpu")
        stop_event = threading.Event()
        stop_event.set()

        samples, result, plies = play_selfplay_game(
            mcts=mcts,
            num_simulations=5,
            max_plies=4,
            pgn_path=pgn_file,
            stop_event=stop_event,
        )

        assert samples == []
        assert result == ""
        assert plies == 0
        assert not pgn_file.exists()


def test_arena_instant_stop():
    net1 = create_model(6, 64, device="cpu")
    net2 = create_model(6, 64, device="cpu")
    stop_event = threading.Event()
    stop_event.set()

    promoted, win_rate, records = evaluate_candidate(
        candidate_net=net1,
        champion_net=net2,
        num_games=5,
        num_simulations=5,
        threshold=0.5,
        device="cpu",
        stop_event=stop_event,
    )

    assert not promoted
    assert len(records) == 0
