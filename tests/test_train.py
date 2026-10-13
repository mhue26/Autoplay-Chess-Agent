"""Tests for training components: replay buffer, self-play, trainer, arena."""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import torch

from rl_chess.core.encoding import ACTION_SPACE_SIZE, NUM_INPUT_CHANNELS
from rl_chess.mcts.search import MCTS
from rl_chess.net.model import create_model
from rl_chess.train.arena import evaluate_candidate
from rl_chess.train.replay import ReplayBuffer
from rl_chess.train.selfplay import play_selfplay_game
from rl_chess.train.trainer import Trainer


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
        model = create_model("tiny", device="cpu")
        trainer = Trainer(model, learning_rate=0.01, device="cpu")
        metrics = trainer.train_step(buf2, batch_size=4)
        assert metrics["loss_total"] > 0.0
        assert metrics["loss_policy"] > 0.0


def test_selfplay_short_game():
    with tempfile.TemporaryDirectory() as tmp_dir:
        pgn_file = Path(tmp_dir) / "test.pgn"
        net = create_model("tiny", device="cpu")
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
    net1 = create_model("tiny", device="cpu")
    net2 = create_model("tiny", device="cpu")

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
