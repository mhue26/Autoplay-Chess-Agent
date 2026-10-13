"""Tests for MCTS search and live analysis extraction."""

from __future__ import annotations

import chess
import numpy as np

from rl_chess.mcts.search import MCTS
from rl_chess.net.model import create_model


def test_mcts_search_starting_position():
    net = create_model("tiny", device="cpu")
    mcts = MCTS(net, c_puct=1.5, device="cpu")

    board = chess.Board()
    move, pi, analysis = mcts.search(board, num_simulations=30, temperature=1.0)

    # 1. Chosen move must be legal
    assert move in board.legal_moves

    # 2. Pi distribution properties
    assert pi.shape == (4672,)
    assert np.isclose(np.sum(pi), 1.0, atol=1e-4)

    # 3. Analysis output
    assert analysis.total_simulations == 30
    assert len(analysis.candidates) == 20
    assert -1.0 <= analysis.root_value <= 1.0
    assert len(analysis.principal_variation) >= 1
    assert analysis.candidates[0].visits >= 1


def test_mcts_greedy_selection():
    net = create_model("tiny", device="cpu")
    mcts = MCTS(net, c_puct=1.5, device="cpu")

    board = chess.Board()
    move, pi, analysis = mcts.search(board, num_simulations=20, temperature=0.0)

    assert move in board.legal_moves
    assert np.max(pi) == 1.0
    assert analysis.candidates[0].move == move


def test_mcts_tree_reuse():
    net = create_model("tiny", device="cpu")
    mcts = MCTS(net, c_puct=1.5, device="cpu")

    board = chess.Board()
    move, _, _ = mcts.search(board, num_simulations=20, temperature=0.0)

    # Advance root by move
    mcts.advance_root(move)
    board.push(move)

    assert mcts.root is not None
    assert mcts.root.board.fen() == board.fen()


def test_mcts_terminal_state():
    # Scholar's mate: 1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7#
    board = chess.Board()
    for san in ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7#"]:
        board.push_san(san)

    assert board.is_checkmate()

    net = create_model("tiny", device="cpu")
    mcts = MCTS(net, device="cpu")
    move, pi, analysis = mcts.search(board, num_simulations=10)

    assert move == chess.Move.null()
    assert np.sum(pi) == 0.0
