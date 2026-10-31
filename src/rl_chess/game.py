"""Shared game-loop helpers for self-play, arena, and benchmark games."""

from __future__ import annotations

import threading
from typing import Callable, List, Optional, Tuple

import chess

from rl_chess.encoding import NUM_HISTORY_STEPS

PickMove = Callable[[chess.Board, List[chess.Board]], Optional[chess.Move]]
OnMove = Callable[[chess.Board, chess.Move], None]


def result_string(board: chess.Board) -> str:
    """PGN result of a finished (or truncated, counted as drawn) game."""
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return "1/2-1/2"
    return "1-0" if outcome.winner == chess.WHITE else "0-1"


def termination_reason(board: chess.Board, plies: int, max_plies: int) -> str:
    """Human-readable explanation of why the game concluded."""
    if plies >= max_plies:
        return f"Max plies reached ({max_plies})"
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        return f"Checkmate ({winner} wins)"
    if board.is_stalemate():
        return "Stalemate"
    if board.is_insufficient_material():
        return "Insufficient material"
    if board.is_seventyfive_moves():
        return "75-move rule"
    if board.is_fivefold_repetition():
        return "Fivefold repetition"
    if board.can_claim_threefold_repetition():
        return "Threefold repetition"
    if board.can_claim_fifty_moves():
        return "Fifty-move rule"
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        return outcome.termination.name.replace("_", " ").title()
    return "Game drawn"


_PIECE_MATERIAL_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def material_balance_white(board: chess.Board) -> int:
    """Signed material difference (White minus Black) using standard piece values."""
    white = 0
    black = 0
    for piece in board.piece_map().values():
        value = _PIECE_MATERIAL_VALUE.get(piece.piece_type, 0)
        if piece.color == chess.WHITE:
            white += value
        else:
            black += value
    return white - black


def format_material_balance(board: chess.Board) -> str:
    """Short label for UI: which side is ahead and by how much."""
    balance = material_balance_white(board)
    if balance == 0:
        return "Material: even"
    if balance > 0:
        return f"White +{balance}"
    return f"Black +{-balance}"


def play_game(
    pick_move: PickMove,
    max_plies: int,
    on_move: Optional[OnMove] = None,
    stop_event: Optional[threading.Event] = None,
) -> Tuple[chess.Board, int]:
    """Play from the start position until the game ends, a bad move is returned, or max_plies.

    ``pick_move(board, history)`` returns the next move; ``on_move(board, move)`` is called
    just before the move is pushed. Returns the final board and the number of plies played.
    """
    board = chess.Board()
    history: List[chess.Board] = []
    ply = 0
    while not board.is_game_over() and ply < max_plies:
        if stop_event and stop_event.is_set():
            break
        move = pick_move(board, history)
        if stop_event and stop_event.is_set():
            break
        if move is None or move not in board.legal_moves:
            break
        history.append(board.copy())
        del history[:-(NUM_HISTORY_STEPS - 1)]
        if on_move:
            on_move(board, move)
        board.push(move)
        ply += 1
    return board, ply
