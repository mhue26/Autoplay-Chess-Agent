"""Human-vs-agent game state: board, history, click-to-move selection, and piece dragging."""

from __future__ import annotations

from typing import List, Optional, Tuple

import chess

from rl_chess.encoding import NUM_HISTORY_STEPS


PROMOTION_PIECES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]


def promotion_choices(board: chess.Board, from_sq: chess.Square, to_sq: chess.Square) -> List[chess.PieceType]:
    """Legal promotion piece types (Queen, Rook, Bishop, Knight) for from_sq -> to_sq."""
    return [p for p in PROMOTION_PIECES if chess.Move(from_sq, to_sq, promotion=p) in board.legal_moves]


def resolve_human_move(board: chess.Board, from_sq: chess.Square, to_sq: chess.Square) -> Optional[chess.Move]:
    """Legal non-promotion move from ``from_sq`` to ``to_sq``."""
    if from_sq == to_sq:
        return None
    plain = chess.Move(from_sq, to_sq)
    if plain in board.legal_moves:
        return plain
    return None


class PlaySession:
    def __init__(self, human_color: chess.Color, drag_threshold: int = 10):
        self.human_color = human_color
        self.drag_threshold = drag_threshold
        self.flipped = human_color == chess.BLACK
        self.pending_promotion: Optional[Tuple[chess.Square, chess.Square]] = None
        self.reset()

    def reset(self) -> None:
        self.board = chess.Board()
        self.history: List[chess.Board] = []
        self.move_sans: List[str] = []
        self.last_move: Optional[chess.Move] = None
        self.selected_sq: Optional[chess.Square] = None
        self.legal_destinations: List[chess.Square] = []
        self.pending_promotion = None
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
        try:
            san = self.board.san(move)
        except Exception:
            san = move.uci()
        self.move_sans.append(san)
        self.history.append(self.board.copy())
        del self.history[:-(NUM_HISTORY_STEPS - 1)]
        self.board.push(move)
        self.last_move = move
        self.pending_promotion = None
        self._select(None)

    def get_move_sans(self) -> List[str]:
        """Return the list of move SANs, rebuilding from move_stack if out of sync."""
        if len(self.move_sans) == len(self.board.move_stack):
            return self.move_sans
        b = chess.Board()
        sans: List[str] = []
        for m in self.board.move_stack:
            try:
                sans.append(b.san(m))
            except Exception:
                sans.append(m.uci())
            b.push(m)
        self.move_sans = sans
        return self.move_sans

    def is_human_turn(self) -> bool:
        return (
            not self.board.is_game_over()
            and self.board.turn == self.human_color
            and self.pending_promotion is None
        )

    def cancel_promotion(self) -> None:
        self.pending_promotion = None
        self._select(None)

    def choose_promotion(self, piece_type: chess.PieceType) -> Optional[chess.Move]:
        if self.pending_promotion is None:
            return None
        from_sq, to_sq = self.pending_promotion
        move = chess.Move(from_sq, to_sq, promotion=piece_type)
        if move in self.board.legal_moves:
            self.pending_promotion = None
            self._select(None)
            return move
        return None

    # -- input -----------------------------------------------------------
    # Each handler returns the move the human completed, if any; the caller applies it.

    def _click(self, sq: chess.Square) -> Optional[chess.Move]:
        if self.selected_sq is not None:
            choices = promotion_choices(self.board, self.selected_sq, sq)
            if choices:
                self.pending_promotion = (self.selected_sq, sq)
                return None
            move = resolve_human_move(self.board, self.selected_sq, sq)
            if move is not None:
                return move
        piece = self.board.piece_at(sq)
        self._select(sq if piece and piece.color == self.human_color else None)
        return None

    def mouse_down(self, pos: Tuple[int, int], sq: Optional[chess.Square]) -> Optional[chess.Move]:
        if self.pending_promotion is not None:
            return None
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
        if self.pending_promotion is not None:
            return
        if self.drag_from_sq is None or self._press_pos is None:
            return
        self.drag_pos = pos
        if not self.is_dragging:
            dx, dy = pos[0] - self._press_pos[0], pos[1] - self._press_pos[1]
            self.is_dragging = dx * dx + dy * dy >= self.drag_threshold ** 2

    def mouse_up(self, sq: Optional[chess.Square]) -> Optional[chess.Move]:
        if self.pending_promotion is not None:
            return None
        if self.drag_from_sq is None:
            return None
        origin, dragged = self.drag_from_sq, self.is_dragging
        self._clear_drag()
        if sq is None:
            return None
        if dragged or sq != origin:
            choices = promotion_choices(self.board, origin, sq)
            if choices:
                self.pending_promotion = (origin, sq)
                self.selected_sq = origin
                return None
            move = resolve_human_move(self.board, origin, sq)
            if move is not None:
                return move
        if not dragged and sq != origin:
            return self._click(sq)
        return None
