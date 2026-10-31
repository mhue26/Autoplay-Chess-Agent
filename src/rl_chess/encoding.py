"""AlphaZero chess board and move encoding.

Action space: 8x8x73 = 4672 actions, indexed as ``from_square * 73 + plane``.
- Planes 0..55:  queen-like moves (8 directions x 7 distances)
- Planes 56..63: knight jumps
- Planes 64..72: underpromotions to knight/bishop/rook (3 file offsets each)
Queen promotions are ordinary queen-like moves onto the back rank.

All moves and boards are oriented to the side to move (Black's view is mirrored).
"""

from __future__ import annotations

from typing import List, Optional

import chess
import numpy as np

NUM_MOVE_TYPES = 73
ACTION_SPACE_SIZE = 64 * NUM_MOVE_TYPES  # 4672

# (d_rank, d_file): N, NE, E, SE, S, SW, W, NW
QUEEN_DIRECTIONS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
KNIGHT_DELTAS = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
# Underpromotion piece -> first plane (offset by file delta -1, 0, +1)
UNDERPROMOTIONS = {chess.KNIGHT: 64, chess.BISHOP: 67, chess.ROOK: 70}

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

# 4 history steps x (12 piece + 2 repetition planes) + 7 global planes
NUM_HISTORY_STEPS = 4
NUM_INPUT_CHANNELS = NUM_HISTORY_STEPS * 14 + 7  # 63


def mirror_square(sq: chess.Square) -> chess.Square:
    """Flip the rank (Black's perspective); self-inverse."""
    return chess.square(chess.square_file(sq), 7 - chess.square_rank(sq))


def orient_move(move: chess.Move, turn: chess.Color) -> chess.Move:
    """Convert between absolute and side-to-move coordinates (self-inverse)."""
    if turn == chess.WHITE:
        return move
    return chess.Move(mirror_square(move.from_square), mirror_square(move.to_square), move.promotion)


def _move_to_plane(move: chess.Move) -> int:
    d_rank = chess.square_rank(move.to_square) - chess.square_rank(move.from_square)
    d_file = chess.square_file(move.to_square) - chess.square_file(move.from_square)

    if move.promotion in UNDERPROMOTIONS:
        return UNDERPROMOTIONS[move.promotion] + d_file + 1
    if (d_rank, d_file) in KNIGHT_DELTAS:
        return 56 + KNIGHT_DELTAS.index((d_rank, d_file))
    for dir_idx, (dr, df) in enumerate(QUEEN_DIRECTIONS):
        for dist in range(1, 8):
            if (d_rank, d_file) == (dr * dist, df * dist):
                return dir_idx * 7 + dist - 1
    raise ValueError(f"Unrecognized move geometry: {move}")


def _plane_to_move(from_sq: chess.Square, plane: int, is_pawn: bool) -> chess.Move:
    rank, file = chess.square_rank(from_sq), chess.square_file(from_sq)

    if plane >= 64:
        for piece, base in UNDERPROMOTIONS.items():
            if base <= plane < base + 3:
                return chess.Move(from_sq, chess.square(file + plane - base - 1, 7), promotion=piece)
        raise ValueError(f"Invalid underpromotion plane: {plane}")
    if plane >= 56:
        dr, df = KNIGHT_DELTAS[plane - 56]
        return chess.Move(from_sq, chess.square(file + df, rank + dr))

    dr, df = QUEEN_DIRECTIONS[plane // 7]
    dist = plane % 7 + 1
    to_rank = rank + dr * dist
    promotion = chess.QUEEN if is_pawn and to_rank == 7 else None
    return chess.Move(from_sq, chess.square(file + df * dist, to_rank), promotion=promotion)


def encode_move(move: chess.Move, turn: chess.Color) -> int:
    """Encode an absolute move as an action index in [0, 4671]."""
    oriented = orient_move(move, turn)
    return oriented.from_square * NUM_MOVE_TYPES + _move_to_plane(oriented)


def decode_action(action: int, board: chess.Board) -> chess.Move:
    """Decode an action index into an absolute move for the side to move on ``board``."""
    from_sq, plane = divmod(action, NUM_MOVE_TYPES)
    abs_from = mirror_square(from_sq) if board.turn == chess.BLACK else from_sq
    is_pawn = board.piece_type_at(abs_from) == chess.PAWN
    return orient_move(_plane_to_move(from_sq, plane, is_pawn), board.turn)


def get_legal_action_mask(board: chess.Board) -> np.ndarray:
    """Boolean mask of shape (4672,) marking legal actions."""
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
    for move in board.legal_moves:
        mask[encode_move(move, board.turn)] = True
    return mask


def encode_board(board: chess.Board, history: Optional[List[chess.Board]] = None) -> np.ndarray:
    """Encode a position as a (NUM_INPUT_CHANNELS, 8, 8) float32 tensor from the mover's view."""
    planes = np.zeros((NUM_INPUT_CHANNELS, 8, 8), dtype=np.float32)
    me = board.turn

    boards = list(history[-(NUM_HISTORY_STEPS - 1):]) if history else []
    boards.append(board)
    while len(boards) < NUM_HISTORY_STEPS:
        boards.insert(0, boards[0])

    ch = 0
    for b in boards:
        for color in (me, not me):
            for piece_type in PIECE_TYPES:
                for sq in b.pieces(piece_type, color):
                    rank = chess.square_rank(sq)
                    planes[ch, rank if me == chess.WHITE else 7 - rank, chess.square_file(sq)] = 1.0
                ch += 1
        repetitions = b.is_repetition(2) + b.is_repetition(3)
        planes[ch] = float(repetitions >= 1)
        planes[ch + 1] = float(repetitions >= 2)
        ch += 2

    planes[ch] = float(me == chess.WHITE)
    planes[ch + 1] = min(1.0, board.fullmove_number / 100.0)
    ch += 2
    for color in (me, not me):
        planes[ch] = float(board.has_kingside_castling_rights(color))
        planes[ch + 1] = float(board.has_queenside_castling_rights(color))
        ch += 2
    planes[ch] = min(1.0, board.halfmove_clock / 100.0)
    ch += 1

    assert ch == NUM_INPUT_CHANNELS
    return planes
