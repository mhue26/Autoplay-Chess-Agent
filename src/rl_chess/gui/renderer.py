"""Board, pieces, analysis panel, and dashboard-tab rendering for Pygame."""

from __future__ import annotations

import math
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chess
import pygame

from rl_chess.gui.widgets import (
    COLOR_BORDER,
    COLOR_PANEL,
    COLOR_PANEL_CARD,
    COLOR_PRIMARY,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
)
from rl_chess.mcts import CandidateMoveAnalysis, MCTSAnalysis

Color = Tuple[int, ...]

COLOR_LIGHT_SQ = (240, 217, 181)
COLOR_DARK_SQ = (181, 136, 99)
COLOR_SELECTED = (205, 210, 106, 180)
COLOR_LAST_MOVE = (205, 210, 106, 120)
COLOR_LEGAL_MARK = (100, 110, 64)
COLOR_BAR_WHITE = (235, 235, 235)
COLOR_BAR_BLACK = (50, 50, 50)
ARROW_COLORS = [(52, 168, 83), (66, 133, 244)]  # top-1, top-2 candidates
COLOR_PIECE_WHITE = (252, 252, 252)
COLOR_PIECE_WHITE_STROKE = (40, 40, 40)
COLOR_PIECE_BLACK = (24, 24, 24)

# Filled glyphs (U+265A-265F); the outline set looks hollow in most fonts.
PIECE_GLYPHS = {
    chess.KING: "\u265a",
    chess.QUEEN: "\u265b",
    chess.ROOK: "\u265c",
    chess.BISHOP: "\u265d",
    chess.KNIGHT: "\u265e",
    chess.PAWN: "\u265f",
}


def _ui_sans_font() -> str:
    available = pygame.font.get_fonts()
    if sys.platform == "darwin":
        for name in ("helveticaneue", "sfprotext", "sfprodisplay", "helvetica"):
            if name in available:
                return name
    return "helvetica" if "helvetica" in available else "arial"


class ChessRenderer:
    def __init__(
        self,
        board_size: int = 560,
        board_offset_x: int = 20,
        board_offset_y: int = 50,
        panel_x: int = 600,
        panel_width: int = 420,
        ui_scale: float = 1.0,
    ):
        self.ui_scale = max(1.0, ui_scale)
        s = self.s
        self.board_size = s(board_size)
        self.board_offset_x = s(board_offset_x)
        self.board_offset_y = s(board_offset_y)
        self.panel_x = s(panel_x)
        self.panel_width = s(panel_width)
        self.sq_size = self.board_size // 8

        pygame.font.init()
        sans = _ui_sans_font()
        glyph_font = "arialunicode" if "arialunicode" in pygame.font.get_fonts() else "Arial"
        self.piece_font = pygame.font.SysFont(glyph_font, int(self.sq_size * 0.75))
        self.font_title = pygame.font.SysFont(sans, s(22), bold=True)
        self.font_lg = pygame.font.SysFont(sans, s(17), bold=True)
        self.font_md = pygame.font.SysFont(sans, s(14))
        self.font_sm = pygame.font.SysFont(sans, s(13))
        self.font_mono = pygame.font.SysFont("menlo" if "menlo" in pygame.font.get_fonts() else "courier", s(13))

    def s(self, value: float) -> int:
        """Scale a design-pixel value to window pixels."""
        return int(round(value * self.ui_scale))

    def rect(self, x: float, y: float, w: float, h: float) -> Tuple[int, int, int, int]:
        return (self.s(x), self.s(y), self.s(w), self.s(h))

    # -- helpers ---------------------------------------------------------

    def _text(self, surface: pygame.Surface, font: pygame.font.Font, text: str, color: Color, x: int, y: int) -> None:
        surface.blit(font.render(text, True, color), (x, y))

    def _card(self, surface: pygame.Surface, rect: pygame.Rect, fill: Color = COLOR_PANEL_CARD, radius: float = 8) -> None:
        pygame.draw.rect(surface, fill, rect, border_radius=self.s(radius))
        pygame.draw.rect(surface, COLOR_BORDER, rect, width=1, border_radius=self.s(radius))

    # -- board geometry --------------------------------------------------

    def square_to_rect(self, sq: chess.Square, flipped: bool = False) -> pygame.Rect:
        file, rank = chess.square_file(sq), chess.square_rank(sq)
        col = 7 - file if flipped else file
        row = rank if flipped else 7 - rank
        return pygame.Rect(
            self.board_offset_x + col * self.sq_size,
            self.board_offset_y + row * self.sq_size,
            self.sq_size,
            self.sq_size,
        )

    def pos_to_square(self, pos: Tuple[int, int], flipped: bool = False) -> Optional[chess.Square]:
        rel_x = pos[0] - self.board_offset_x
        rel_y = pos[1] - self.board_offset_y
        if not (0 <= rel_x < self.board_size and 0 <= rel_y < self.board_size):
            return None
        col, row = rel_x // self.sq_size, rel_y // self.sq_size
        return chess.square(7 - col if flipped else col, row if flipped else 7 - row)

    # -- board -----------------------------------------------------------

    def _blit_piece(self, surface: pygame.Surface, center: Tuple[int, int], piece: chess.Piece) -> None:
        char = PIECE_GLYPHS[piece.piece_type]
        cx, cy = center
        if piece.color == chess.WHITE:
            stroke = self.piece_font.render(char, True, COLOR_PIECE_WHITE_STROKE)
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                surface.blit(stroke, stroke.get_rect(center=(cx + dx, cy + dy)))
            fill = self.piece_font.render(char, True, COLOR_PIECE_WHITE)
        else:
            shadow = self.piece_font.render(char, True, (10, 10, 10))
            surface.blit(shadow, shadow.get_rect(center=(cx + 1, cy + 1)))
            fill = self.piece_font.render(char, True, COLOR_PIECE_BLACK)
        surface.blit(fill, fill.get_rect(center=center))

    def _highlight(self, surface: pygame.Surface, color: Color, sq: chess.Square, flipped: bool) -> None:
        overlay = pygame.Surface((self.sq_size, self.sq_size), pygame.SRCALPHA)
        overlay.fill(color)
        surface.blit(overlay, self.square_to_rect(sq, flipped))

    def draw_board(
        self,
        surface: pygame.Surface,
        board: chess.Board,
        flipped: bool = False,
        selected_sq: Optional[chess.Square] = None,
        last_move: Optional[chess.Move] = None,
        legal_destinations: Optional[List[chess.Square]] = None,
        analysis: Optional[MCTSAnalysis] = None,
        drag_from_sq: Optional[chess.Square] = None,
        drag_piece: Optional[chess.Piece] = None,
        drag_pos: Optional[Tuple[int, int]] = None,
    ) -> None:
        for r in range(8):
            for c in range(8):
                color = COLOR_LIGHT_SQ if (r + c) % 2 == 0 else COLOR_DARK_SQ
                rect = (self.board_offset_x + c * self.sq_size, self.board_offset_y + r * self.sq_size, self.sq_size, self.sq_size)
                pygame.draw.rect(surface, color, rect)

        if last_move is not None:
            self._highlight(surface, COLOR_LAST_MOVE, last_move.from_square, flipped)
            self._highlight(surface, COLOR_LAST_MOVE, last_move.to_square, flipped)
        if selected_sq is not None:
            self._highlight(surface, COLOR_SELECTED, selected_sq, flipped)

        for dest in legal_destinations or []:
            center = self.square_to_rect(dest, flipped).center
            if board.piece_at(dest):
                pygame.draw.circle(surface, COLOR_LEGAL_MARK, center, int(self.sq_size * 0.42), 4)
            else:
                pygame.draw.circle(surface, COLOR_LEGAL_MARK, center, int(self.sq_size * 0.16))

        if analysis and analysis.candidates:
            self._draw_candidate_arrows(surface, analysis.candidates[:2], flipped)

        for sq, piece in board.piece_map().items():
            if drag_piece is not None and sq == drag_from_sq:
                continue
            self._blit_piece(surface, self.square_to_rect(sq, flipped).center, piece)
        if drag_piece is not None and drag_pos is not None:
            self._blit_piece(surface, drag_pos, drag_piece)

        for i in range(8):
            label_color = COLOR_DARK_SQ if i % 2 == 0 else COLOR_LIGHT_SQ
            rank_label = str(i + 1 if flipped else 8 - i)
            file_label = chr(ord("a") + (7 - i if flipped else i))
            self._text(surface, self.font_sm, rank_label, label_color,
                       self.board_offset_x + self.s(4), self.board_offset_y + i * self.sq_size + self.s(4))
            self._text(surface, self.font_sm, file_label, label_color,
                       self.board_offset_x + (i + 1) * self.sq_size - self.s(12),
                       self.board_offset_y + self.board_size - self.s(16))

    def _draw_candidate_arrows(
        self, surface: pygame.Surface, candidates: Sequence[CandidateMoveAnalysis], flipped: bool
    ) -> None:
        for cand, color in zip(candidates, ARROW_COLORS):
            start = self.square_to_rect(cand.move.from_square, flipped).center
            end = self.square_to_rect(cand.move.to_square, flipped).center
            pygame.draw.line(surface, color, start, end, width=4)
            angle = math.atan2(end[1] - start[1], end[0] - start[0])
            head = [
                (end[0] - 12 * math.cos(angle + d), end[1] - 12 * math.sin(angle + d))
                for d in (-math.pi / 6, math.pi / 6)
            ]
            pygame.draw.polygon(surface, color, [end, *head])

    def draw_game_end_overlay(
        self,
        surface: pygame.Surface,
        board: chess.Board,
        notice: Any,
        elapsed_ms: int = 0,
        flipped: bool = False,
    ) -> None:
        """Render banner overlay over the board when the game is completed."""
        s = self.s
        card = pygame.Rect(self.board_offset_x + s(40), self.board_offset_y + s(200), self.board_size - s(80), s(120))
        overlay = pygame.Surface((card.width, card.height), pygame.SRCALPHA)
        overlay.fill((20, 22, 25, 235))
        surface.blit(overlay, (card.x, card.y))
        pygame.draw.rect(surface, COLOR_BORDER, card, width=1, border_radius=s(8))
        tone = getattr(notice, "tone", "draw")
        tone_color = COLOR_PRIMARY if tone == "win" else ((220, 53, 69) if tone == "loss" else COLOR_TEXT)
        self._text(surface, self.font_lg, getattr(notice, "title", "Game Over"), tone_color, card.x + s(20), card.y + s(25))
        self._text(surface, self.font_md, getattr(notice, "subtitle", ""), COLOR_TEXT_MUTED, card.x + s(20), card.y + s(65))

    # -- play tab side panel --------------------------------------------

    def draw_activity_strip(self, surface: pygame.Surface, message: str, busy: bool = False) -> None:
        """Global status line under the tab bar (training, loading weights, MCTS, etc.)."""
        s = self.s
        rect = pygame.Rect(0, s(44), s(1040), s(24))
        pygame.draw.rect(surface, (24, 26, 30), rect)
        pygame.draw.line(surface, COLOR_BORDER, (rect.x, rect.bottom - 1), (rect.right, rect.bottom - 1), 1)

        dot_x = rect.x + s(14)
        dot_y = rect.centery
        if busy:
            pulse = 0.45 + 0.55 * math.sin(pygame.time.get_ticks() / 220.0)
            radius = int(s(4) + s(2) * pulse)
            pygame.draw.circle(surface, COLOR_PRIMARY, (dot_x, dot_y), radius)
        else:
            pygame.draw.circle(surface, COLOR_PRIMARY, (dot_x, dot_y), s(4))

        label = message
        if busy:
            phase = (pygame.time.get_ticks() // 400) % 4
            label = f"{message}{'.' * phase}"
        self._text(surface, self.font_sm, label, COLOR_TEXT, rect.x + s(28), rect.y + s(5))

    def draw_analysis_panel(
        self,
        surface: pygame.Surface,
        board: chess.Board,
        analysis: Optional[MCTSAnalysis],
        agent_name: str,
        is_thinking: bool,
        weights_status: str = "",
        weights_loading: bool = False,
    ) -> None:
        s = self.s
        panel = pygame.Rect(self.panel_x, self.board_offset_y, self.panel_width, self.board_size)
        self._card(surface, panel, COLOR_PANEL)

        x = self.panel_x + s(16)
        y = self.board_offset_y + s(16)
        self._text(surface, self.font_lg, "Live Analysis", COLOR_TEXT, x, y)
        y += s(26)

        turn = "Agent thinking..." if is_thinking else ("White to move" if board.turn == chess.WHITE else "Black to move")
        self._text(surface, self.font_md, f"Turn: {turn}", COLOR_PRIMARY if is_thinking else COLOR_TEXT_MUTED, x, y)
        y += s(22)
        self._text(surface, self.font_sm, f"Agent: {agent_name}", COLOR_TEXT_MUTED, x, y)
        y += s(22)

        if weights_loading:
            w_color = COLOR_PRIMARY
            w_label = weights_status or "Loading weights..."
        elif weights_status:
            w_color = COLOR_PRIMARY if "updated" in weights_status.lower() else COLOR_TEXT_MUTED
            w_label = weights_status
        else:
            w_color = COLOR_TEXT_MUTED
            w_label = "Weights: in memory"
        self._text(surface, self.font_sm, f"Weights: {w_label}", w_color, x, y)
        y += s(24)

        # Win-expectancy bar (white's perspective)
        root_val = analysis.root_value if analysis else 0.0
        white_pct = ((root_val if board.turn == chess.WHITE else -root_val) + 1.0) / 2.0
        self._text(surface, self.font_md, f"Win Expectancy: {white_pct * 100:.1f}% White", COLOR_TEXT, x, y)
        y += s(20)
        bar_w, bar_h = self.panel_width - 2 * s(16), s(16)
        pygame.draw.rect(surface, COLOR_BAR_BLACK, (x, y, bar_w, bar_h), border_radius=s(4))
        if int(bar_w * white_pct) > 0:
            pygame.draw.rect(surface, COLOR_BAR_WHITE, (x, y, int(bar_w * white_pct), bar_h), border_radius=s(4))
        y += s(26)

        # Candidate table
        card = pygame.Rect(x, y, bar_w, s(175))
        pygame.draw.rect(surface, COLOR_PANEL_CARD, card, border_radius=s(8))
        tx = self.panel_x + s(24)
        self._text(surface, self.font_md, "Top Candidate Moves (MCTS)", COLOR_TEXT, tx, y + s(10))
        self._text(surface, self.font_sm, "Move        Visits     Share        Q", COLOR_TEXT_MUTED, tx, y + s(32))

        row_y = y + s(52)
        if analysis and analysis.candidates:
            for idx, c in enumerate(analysis.candidates[:4]):
                row = f"#{idx + 1}  {c.san:<8} {c.visits:<8} {c.visit_share * 100:>5.1f}%   {c.q_value:>+5.2f}"
                self._text(surface, self.font_sm, row, COLOR_PRIMARY if idx == 0 else COLOR_TEXT, tx, row_y)
                row_y += s(24)
        else:
            self._text(surface, self.font_sm, "Awaiting search...", COLOR_TEXT_MUTED, tx, row_y)

    # -- dashboard tabs --------------------------------------------------

    def draw_job_header(
        self,
        surface: pygame.Surface,
        title: str,
        status: str,
        info: str = "",
        badge: bool = False,
        secondary: str = "",
    ) -> None:
        """Title row of the Training / Benchmark tabs: title, status, and a champion label or Elo badge."""
        s = self.s
        x, y = s(20), s(54)
        self._text(surface, self.font_title, title, COLOR_TEXT, x, y)
        status_color = COLOR_PRIMARY if status in ("running", "loading") else COLOR_TEXT_MUTED
        self._text(surface, self.font_md, f"Status: {status.upper()}", status_color, x + s(460), y + s(6))
        if secondary:
            self._text(surface, self.font_sm, secondary, COLOR_PRIMARY, x + s(460), y + s(26))
        if not info:
            return
        if badge:
            rect = pygame.Rect(x + s(650), y - s(2), s(330), s(36))
            self._card(surface, rect, COLOR_PANEL_CARD, radius=6)
            pygame.draw.rect(surface, COLOR_PRIMARY, rect, width=1, border_radius=s(6))
            self._text(surface, self.font_md, info, COLOR_PRIMARY, rect.x + s(12), rect.y + s(9))
        else:
            self._text(surface, self.font_md, info, COLOR_TEXT, x + s(650), y + s(6))

    def _draw_table(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        title: str,
        header: str,
        rows: List[Tuple[str, Color]],
        empty_message: str,
    ) -> None:
        s = self.s
        self._card(surface, rect)
        x = rect.x + s(16)
        self._text(surface, self.font_lg, title, COLOR_TEXT, x, rect.y + s(12))
        self._text(surface, self.font_mono, header, COLOR_TEXT_MUTED, x, rect.y + s(42))
        pygame.draw.line(surface, COLOR_BORDER, (x, rect.y + s(62)), (rect.right - s(16), rect.y + s(62)))
        row_y = rect.y + s(70)
        if not rows:
            self._text(surface, self.font_md, empty_message, COLOR_TEXT_MUTED, x, row_y)
        for text, color in rows:
            self._text(surface, self.font_mono, text, color, x, row_y)
            row_y += s(26)

    def draw_stats_tab(
        self,
        surface: pygame.Surface,
        summary: Dict[str, Any],
        elo_history: List[Dict[str, Any]],
        checkpoints: List[Any],
    ) -> None:
        s = self.s
        x, y = s(20), s(54)
        self._text(surface, self.font_title, "Training & Evaluation Ledger", COLOR_TEXT, x, y)
        y += s(40)

        champion = summary.get("champion")
        latest_elo = summary.get("latest_elo")
        metrics = [
            ("Self-Play Games", str(summary.get("selfplay_games", 0)), f"~{summary.get('selfplay_positions_estimated', 0)} positions"),
            ("Human Play", str(summary.get("human_games", 0)), "Play Computer matches"),
            ("Training Steps", str(summary.get("training_steps", 0)), "Optimizer steps logged"),
            ("Checkpoints", str(summary.get("checkpoints_count", 0)), f"Champ: {champion['name'] if champion else 'None'}"),
            ("Latest Elo", f"{latest_elo['elo']:.0f}" if latest_elo else "N/A", "Stockfish Anchor"),
        ]
        card_h, gap = s(75), s(15)
        card_w = (s(1000) - gap * (len(metrics) - 1)) // len(metrics)
        for i, (label, value, sub) in enumerate(metrics):
            cx = x + i * (card_w + gap)
            self._card(surface, pygame.Rect(cx, y, card_w, card_h))
            self._text(surface, self.font_sm, label, COLOR_TEXT_MUTED, cx + s(14), y + s(8))
            self._text(surface, self.font_title, value, COLOR_PRIMARY if label == "Latest Elo" else COLOR_TEXT, cx + s(14), y + s(26))
            self._text(surface, self.font_sm, sub, COLOR_TEXT_MUTED, cx + s(14), y + s(54))
        y += s(95)

        elo_rows = [
            (f"{r.get('checkpoint_name', ''):<14} {r.get('generation', 0):<5} {r.get('elo', 0):<8.0f} "
             f"{'[%.0f, %.0f]' % (r.get('ci_low') or 0, r.get('ci_high') or 0):<14} {r.get('n_games', 0):<6} "
             f"{str(r.get('created_at', ''))[:10]:<10}", COLOR_TEXT)
            for r in elo_history[-11:]
        ]
        self._draw_table(
            surface, pygame.Rect(x, y, s(620), s(390)), "Calibrated Elo History",
            f"{'Checkpoint':<14} {'Gen':<5} {'Elo':<8} {'95% CI':<14} {'Games':<6} {'Date':<10}",
            elo_rows, "No Elo estimates recorded yet. Run a benchmark!",
        )

        cp_rows = [
            (f"{cp.name:<12} {cp.generation:<5} {'Yes (Champ)' if cp.promoted else 'No':<10}",
             COLOR_PRIMARY if cp.promoted else COLOR_TEXT)
            for cp in checkpoints[-11:]
        ]
        self._draw_table(
            surface, pygame.Rect(x + s(640), y, s(360), s(390)), "Registered Checkpoints",
            f"{'Name':<12} {'Gen':<5} {'Promoted':<10}", cp_rows, "No checkpoints saved yet.",
        )
