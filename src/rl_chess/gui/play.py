"""Human-vs-agent game state: board, history, click-to-move selection, and piece dragging."""

from __future__ import annotations

from typing import List, Optional, Tuple

import chess

from rl_chess.encoding import NUM_HISTORY_STEPS


def resolve_human_move(board: chess.Board, from_sq: chess.Square, to_sq: chess.Square) -> Optional[chess.Move]:
    """Legal move for a click or drop from ``from_sq`` to ``to_sq`` (promotions auto-queen)."""
    if from_sq == to_sq:
        return None
    for move in (chess.Move(from_sq, to_sq, promotion=chess.QUEEN), chess.Move(from_sq, to_sq)):
        if move in board.legal_moves:
            return move
    return None


class PlaySession:
    def __init__(self, human_color: chess.Color, drag_threshold: int = 10):
        self.human_color = human_color
        self.drag_threshold = drag_threshold
        self.flipped = human_color == chess.BLACK
        self.reset()

    def reset(self) -> None:
        self.board = chess.Board()
        self.history: List[chess.Board] = []
        self.last_move: Optional[chess.Move] = None
        self.selected_sq: Optional[chess.Square] = None
        self.legal_destinations: List[chess.Square] = []
        self._clear_drag()

    def _clear_drag(self) -> None:
        self.drag_from_sq: Optional[chess.Square] = None
        self.drag_piece: Optional[chess.Piece] = None
        self.drag_pos: Optional[Tuple[int, int]] = None
        self._press_pos: Optional[Tuple[int, int]] = None
        self.is_dragging = False

    def _select(self, sq: Optional[chess.Square]) -> None:
        self.selected_sq = sq
        self.legal_destinations = [] if sq is None else [m.to_square for m in self.board.legal_moves if m.from_square == sq]

    def push(self, move: chess.Move) -> None:
        self.history.append(self.board.copy())
        del self.history[:-(NUM_HISTORY_STEPS - 1)]
        self.board.push(move)
        self.last_move = move
        self._select(None)

    def is_human_turn(self) -> bool:
        return not self.board.is_game_over() and self.board.turn == self.human_color

    # -- input -----------------------------------------------------------
    # Each handler returns the move the human completed, if any; the caller applies it.

    def _click(self, sq: chess.Square) -> Optional[chess.Move]:
        if self.selected_sq is not None:
            move = resolve_human_move(self.board, self.selected_sq, sq)
            if move is not None:
                return move
        piece = self.board.piece_at(sq)
        self._select(sq if piece and piece.color == self.human_color else None)
        return None

    def mouse_down(self, pos: Tuple[int, int], sq: Optional[chess.Square]) -> Optional[chess.Move]:
        if sq is None or not self.is_human_turn():
            return None
        piece = self.board.piece_at(sq)
        if piece and piece.color == self.human_color:
            self.drag_from_sq, self.drag_piece = sq, piece
            self.drag_pos = self._press_pos = pos
            self.is_dragging = False
            self._select(sq)
            return None
        return self._click(sq)

    def mouse_motion(self, pos: Tuple[int, int]) -> None:
        if self.drag_from_sq is None or self._press_pos is None:
            return
        self.drag_pos = pos
        if not self.is_dragging:
            dx, dy = pos[0] - self._press_pos[0], pos[1] - self._press_pos[1]
            self.is_dragging = dx * dx + dy * dy >= self.drag_threshold ** 2

    def mouse_up(self, sq: Optional[chess.Square]) -> Optional[chess.Move]:
        if self.drag_from_sq is None:
            return None
        origin, dragged = self.drag_from_sq, self.is_dragging
        self._clear_drag()
        if sq is None:
            return None
        if dragged or sq != origin:
            move = resolve_human_move(self.board, origin, sq)
            if move is not None:
                return move
        if not dragged and sq != origin:
            return self._click(sq)
        return None
