"""Exhaustive tests for AlphaZero board and move encoding/decoding."""

from __future__ import annotations

import random
import chess
import numpy as np

from rl_chess.encoding import (
    ACTION_SPACE_SIZE,
    NUM_INPUT_CHANNELS,
    decode_action,
    encode_board,
    encode_move,
    get_legal_action_mask,
)


def test_starting_position_moves():
    board = chess.Board()
    legal_moves = list(board.legal_moves)
    assert len(legal_moves) == 20

    mask = get_legal_action_mask(board)
    assert mask.shape == (ACTION_SPACE_SIZE,)
    assert np.sum(mask) == 20

    for move in legal_moves:
        action = encode_move(move, board.turn)
        assert 0 <= action < ACTION_SPACE_SIZE
        assert mask[action] or mask[action] == 1
        decoded = decode_action(action, board)
        assert decoded == move, f"Expected {move}, decoded {decoded}"


def test_promotions_and_underpromotions():
    # White pawn on a7 ready to promote to Q, R, B, N
    # Black king on h8, White king on h1
    board = chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")
    moves = list(board.legal_moves)
    # Expected: a7a8q, a7a8r, a7a8b, a7a8n, and king moves
    promo_moves = [m for m in moves if m.promotion is not None]
    assert len(promo_moves) == 4

    mask = get_legal_action_mask(board)
    for m in promo_moves:
        action = encode_move(m, board.turn)
        assert mask[action]
        decoded = decode_action(action, board=board)
        assert decoded == m

    # Black pawn on a2 ready to promote
    board_b = chess.Board("7k/8/8/8/8/8/p7/7K b - - 0 1")
    moves_b = list(board_b.legal_moves)
    promo_moves_b = [m for m in moves_b if m.promotion is not None]
    assert len(promo_moves_b) == 4

    mask_b = get_legal_action_mask(board_b)
    for m in promo_moves_b:
        action = encode_move(m, board_b.turn)
        assert mask_b[action]
        decoded = decode_action(action, board=board_b)
        assert decoded == m


def test_castling_and_en_passant():
    # Board with White castling rights both sides
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    castle_moves = [m for m in board.legal_moves if board.is_castling(m)]
    assert len(castle_moves) == 2  # e1g1, e1c1
    for m in castle_moves:
        act = encode_move(m, board.turn)
        assert decode_action(act, board) == m

    # Board with Black castling
    board_b = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
    castle_moves_b = [m for m in board_b.legal_moves if board_b.is_castling(m)]
    assert len(castle_moves_b) == 2
    for m in castle_moves_b:
        act = encode_move(m, board_b.turn)
        assert decode_action(act, board_b) == m

    # En passant: White pawn on e5, Black pawn plays d7-d5 -> e5xd6 ep
    ep_board = chess.Board("8/8/8/3pP3/8/8/8/4K2k w - d6 0 1")
    ep_moves = [m for m in ep_board.legal_moves if ep_board.is_en_passant(m)]
    assert len(ep_moves) == 1
    ep_move = ep_moves[0]
    act = encode_move(ep_move, ep_board.turn)
    assert decode_action(act, ep_board) == ep_move


def test_round_trip_random_games():
    """Simulate random games and assert every single legal move round-trips perfectly."""
    random.seed(42)
    num_games = 25
    total_moves_tested = 0

    for _ in range(num_games):
        board = chess.Board()
        while not board.is_game_over() and board.fullmove_number < 50:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            mask = get_legal_action_mask(board)
            assert np.sum(mask) == len(legal_moves)

            for move in legal_moves:
                action = encode_move(move, board.turn)
                assert 0 <= action < ACTION_SPACE_SIZE
                assert mask[action]
                decoded = decode_action(action, board=board)
                assert decoded == move, f"Mismatch: original={move}, decoded={decoded}, fen={board.fen()}"
                total_moves_tested += 1

            # Play a random move to advance position
            board.push(random.choice(legal_moves))

    assert total_moves_tested > 1000, f"Only tested {total_moves_tested} moves"


def test_board_encoding_shape_and_range():
    board = chess.Board()
    planes = encode_board(board)
    assert planes.shape == (NUM_INPUT_CHANNELS, 8, 8)
    assert planes.dtype == np.float32
    assert np.all(planes >= 0.0) and np.all(planes <= 1.0)

    # Test with history
    h1 = board.copy()
    board.push_san("e4")
    h2 = board.copy()
    board.push_san("e5")
    planes_h = encode_board(board, history=[h1, h2])
    assert planes_h.shape == (NUM_INPUT_CHANNELS, 8, 8)
    assert planes_h.dtype == np.float32
