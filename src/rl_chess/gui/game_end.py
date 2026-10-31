"""Human-readable game-over messaging for the Play tab."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import chess


@dataclass(frozen=True)
class GameEndNotice:
    """Banner copy and accent tone for the end-of-game UI."""

    title: str
    subtitle: str
    tone: str  # "win" | "loss" | "draw"


_TERMINATION_LABELS = {
    chess.Termination.CHECKMATE: "Checkmate",
    chess.Termination.STALEMATE: "Stalemate",
    chess.Termination.INSUFFICIENT_MATERIAL: "Insufficient material",
    chess.Termination.SEVENTYFIVE_MOVES: "75-move rule",
    chess.Termination.FIVEFOLD_REPETITION: "Fivefold repetition",
    chess.Termination.FIFTY_MOVES: "Fifty-move rule",
    chess.Termination.THREEFOLD_REPETITION: "Threefold repetition",
    chess.Termination.VARIANT_WIN: "Variant win",
    chess.Termination.VARIANT_LOSS: "Variant loss",
    chess.Termination.VARIANT_DRAW: "Variant draw",
}


def termination_detail(board: chess.Board) -> str:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return "Game over"
    label = _TERMINATION_LABELS.get(outcome.termination)
    if label:
        return label
    return outcome.termination.name.replace("_", " ").title()


def game_end_notice(board: chess.Board, human_color: chess.Color) -> GameEndNotice:
    """Build title/subtitle for a finished human vs agent game."""
    detail = termination_detail(board)
    outcome = board.outcome(claim_draw=True)

    if outcome is None or outcome.winner is None:
        return GameEndNotice(title="Draw", subtitle=detail, tone="draw")

    human_won = outcome.winner == human_color
    if human_won:
        title = "You win!"
        tone = "win"
    else:
        title = "You lose"
        tone = "loss"

    if outcome.termination == chess.Termination.CHECKMATE:
        subtitle = "Checkmate"
    else:
        subtitle = detail

    return GameEndNotice(title=title, subtitle=subtitle, tone=tone)


def checkmated_king_square(board: chess.Board) -> Optional[chess.Square]:
    if not board.is_checkmate():
        return None
    return board.king(board.turn)
