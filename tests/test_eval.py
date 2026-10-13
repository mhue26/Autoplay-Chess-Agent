"""Tests for Bradley-Terry Elo estimation and benchmark harness."""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from rl_chess.eval.elo import MatchResult, expected_score, fit_elo_bradley_terry
from rl_chess.eval.benchmark import OpponentRung, find_stockfish_executable, play_benchmark_game
from rl_chess.mcts.search import MCTS
from rl_chess.net.model import create_model


def test_expected_score():
    # Equal ratings -> 50%
    assert pytest.approx(expected_score(1500, 1500), 0.001) == 0.5
    # 400 Elo advantage -> ~90.9%
    assert pytest.approx(expected_score(1900, 1500), 0.001) == 10 / 11
    # 400 Elo disadvantage -> ~9.1%
    assert pytest.approx(expected_score(1500, 1900), 0.001) == 1 / 11


def test_fit_elo_bradley_terry():
    matches = [
        # Against 1000 Elo: 8 wins, 2 losses (score 8.0/10)
        MatchResult(opponent_name="Opp1000", opponent_elo=1000.0, wins=8, draws=0, losses=2),
        # Against 1400 Elo: 2 wins, 8 losses (score 2.0/10)
        MatchResult(opponent_name="Opp1400", opponent_elo=1400.0, wins=2, draws=0, losses=8),
    ]
    # Expected rating should fall roughly midway around 1200
    estimate = fit_elo_bradley_terry(matches)
    assert 1150.0 <= estimate.rating <= 1250.0
    assert estimate.ci_low < estimate.rating < estimate.ci_high
    assert estimate.total_games == 20
    assert estimate.total_score == 10.0


def test_fit_elo_all_wins_all_losses():
    all_wins = [MatchResult("Opp", 1000.0, wins=10, draws=0, losses=0)]
    est_win = fit_elo_bradley_terry(all_wins)
    assert est_win.rating > 1000.0

    all_losses = [MatchResult("Opp", 1000.0, wins=0, draws=0, losses=10)]
    est_loss = fit_elo_bradley_terry(all_losses)
    assert est_loss.rating < 1000.0


def test_play_benchmark_game_random():
    net = create_model("tiny", device="cpu")
    mcts = MCTS(net, device="cpu")
    rung = OpponentRung("Random-Player", elo=200.0)

    res, plies = play_benchmark_game(
        mcts=mcts,
        engine=None,
        rung=rung,
        agent_is_white=True,
        num_simulations=5,
        max_plies=4,
    )

    assert res in ["1-0", "0-1", "1/2-1/2"]
    assert plies <= 4


def test_find_stockfish():
    sf = find_stockfish_executable()
    # On this machine, Stockfish was installed to /opt/homebrew/bin/stockfish
    assert sf is not None
    assert Path(sf).is_file()
