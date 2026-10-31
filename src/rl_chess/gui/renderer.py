"""Board, pieces, analysis panel, and dashboard-tab rendering for Pygame."""

from __future__ import annotations

import math
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chess
import pygame

from rl_chess.game import format_material_balance, material_balance_white
from rl_chess.gui.game_end import checkmated_king_square
from rl_chess.gui.widgets import (
    COLOR_BORDER,
    COLOR_DANGER,
    COLOR_PANEL,
    COLOR_PANEL_CARD,
    COLOR_PRIMARY,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
    _fit_text,
)
from rl_chess.mcts import CandidateMoveAnalysis, MCTSAnalysis

Color = Tuple[int, ...]

COLOR_POPUP_BG = (24, 26, 32)
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

    def game_end_card_rect(self) -> pygame.Rect:
        """Modal card rectangle centered on the board."""
        s = self.s
        card_w, card_h = s(320), s(180)
        card_x = self.board_offset_x + (self.board_size - card_w) // 2
        card_y = self.board_offset_y + (self.board_size - card_h) // 2
        return pygame.Rect(card_x, card_y, card_w, card_h)

    def game_end_button_rect(self) -> pygame.Rect:
        """New Game button rectangle inside the endgame modal card."""
        s = self.s
        card = self.game_end_card_rect()
        btn_w, btn_h = s(140), s(38)
        btn_x = card.centerx - btn_w // 2
        btn_y = card.y + s(114)
        return pygame.Rect(btn_x, btn_y, btn_w, btn_h)

    def draw_game_end_overlay(
        self,
        surface: pygame.Surface,
        board: chess.Board,
        notice: Any,
        elapsed_ms: int = 0,
        flipped: bool = False,
        new_game_btn: Optional[Any] = None,
    ) -> None:
        """Render dimmed board and modal card popup over the board when the game is completed."""
        s = self.s

        # 1. On checkmate, highlight the mated king with checkmated_king_square before the dim
        mated_sq = checkmated_king_square(board)
        if mated_sq is not None:
            self._highlight(surface, (220, 53, 69, 180), mated_sq, flipped)
            piece = board.piece_at(mated_sq)
            if piece is not None:
                self._blit_piece(surface, self.square_to_rect(mated_sq, flipped).center, piece)

        # 2. Dim the board
        dim = pygame.Surface((self.board_size, self.board_size), pygame.SRCALPHA)
        dim.fill((10, 12, 16, 160))
        surface.blit(dim, (self.board_offset_x, self.board_offset_y))

        # 3. Center a card on the board
        card = self.game_end_card_rect()
        pygame.draw.rect(surface, COLOR_POPUP_BG, card, border_radius=s(8))
        pygame.draw.rect(surface, COLOR_BORDER, card, width=1, border_radius=s(8))

        # 4. Title and termination subtitle
        tone = getattr(notice, "tone", "draw")
        tone_color = COLOR_PRIMARY if tone == "win" else (COLOR_DANGER if tone == "loss" else COLOR_TEXT)
        title = getattr(notice, "title", "Game Over")
        subtitle = getattr(notice, "subtitle", "")

        title_surf = self.font_title.render(title, True, tone_color)
        surface.blit(title_surf, title_surf.get_rect(center=(card.centerx, card.y + s(38))))

        if subtitle:
            sub_surf = _fit_text(self.font_md, subtitle, card.width - s(32), COLOR_TEXT_MUTED)
            surface.blit(sub_surf, sub_surf.get_rect(center=(card.centerx, card.y + s(74))))

        # 5. Render New Game button if provided
        if new_game_btn is not None:
            new_game_btn.render(surface, self.font_md)

    def promotion_rects(
        self, to_sq: chess.Square, flipped: bool = False
    ) -> List[Tuple[pygame.Rect, chess.PieceType]]:
        """Return (rect, piece_type) pairs for the vertical promotion picker on to_sq's file."""
        dest_rect = self.square_to_rect(to_sq, flipped)
        rel_y = dest_rect.y - self.board_offset_y
        row = rel_y // self.sq_size
        step = 1 if row < 4 else -1
        pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        return [
            (
                pygame.Rect(
                    dest_rect.x,
                    dest_rect.y + i * step * self.sq_size,
                    self.sq_size,
                    self.sq_size,
                ),
                p,
            )
            for i, p in enumerate(pieces)
        ]

    def promotion_choice_at(
        self, pos: Tuple[int, int], to_sq: chess.Square, flipped: bool = False
    ) -> Optional[chess.PieceType]:
        """Return which promotion piece was clicked, or None if outside the picker."""
        for rect, piece_type in self.promotion_rects(to_sq, flipped):
            if rect.collidepoint(pos):
                return piece_type
        return None

    def draw_promotion_picker(
        self,
        surface: pygame.Surface,
        to_sq: chess.Square,
        color: chess.Color,
        flipped: bool = False,
        hover_pos: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Render the 4-choice vertical promotion picker popup on the board."""
        rects = self.promotion_rects(to_sq, flipped)
        if not rects:
            return

        min_y = min(r.y for r, _ in rects)
        max_y = max(r.bottom for r, _ in rects)
        panel_rect = pygame.Rect(rects[0][0].x, min_y, self.sq_size, max_y - min_y)

        # Semi-transparent shadow around the panel
        shadow_rect = panel_rect.inflate(self.s(8), self.s(8))
        shadow_surf = pygame.Surface((shadow_rect.width, shadow_rect.height), pygame.SRCALPHA)
        shadow_surf.fill((0, 0, 0, 110))
        surface.blit(shadow_surf, shadow_rect)

        # Panel background: dark for white pieces, light for black pieces for maximum contrast
        bg_color = (36, 40, 48) if color == chess.WHITE else (235, 238, 242)
        pygame.draw.rect(surface, bg_color, panel_rect, border_radius=self.s(6))
        pygame.draw.rect(surface, COLOR_PRIMARY, panel_rect, width=self.s(2), border_radius=self.s(6))

        for rect, piece_type in rects:
            if hover_pos and rect.collidepoint(hover_pos):
                hover_bg = (56, 62, 74) if color == chess.WHITE else (210, 215, 222)
                pygame.draw.rect(surface, hover_bg, rect, border_radius=self.s(4))
            self._blit_piece(surface, rect.center, chess.Piece(piece_type, color))

        # Separators between choice squares
        for i in range(len(rects) - 1):
            r1, _ = rects[i]
            r2, _ = rects[i + 1]
            sep_y = r1.bottom if r1.y < r2.y else r1.y
            sep_color = (60, 65, 75) if color == chess.WHITE else (200, 205, 212)
            pygame.draw.line(
                surface,
                sep_color,
                (panel_rect.x + self.s(4), sep_y),
                (panel_rect.right - self.s(4), sep_y),
                1,
            )

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

    def play_moves_card_rect(self) -> pygame.Rect:
        """Card rectangle for Move List in the Live Analysis panel."""
        s = self.s
        bar_w = self.panel_width - 2 * s(16)
        x = self.panel_x + s(16)
        y = self.board_offset_y + s(310)
        h = self.board_size - s(324)
        return pygame.Rect(x, y, bar_w, h)

    def play_moves_visible_rows(self) -> int:
        """Number of full move rows visible inside the Move List card."""
        s = self.s
        card = self.play_moves_card_rect()
        clip_h = card.height - s(36)
        return max(1, clip_h // s(20))

    def draw_analysis_panel(
        self,
        surface: pygame.Surface,
        board: chess.Board,
        analysis: Optional[MCTSAnalysis],
        agent_name: str,
        is_thinking: bool,
        weights_status: str = "",
        weights_loading: bool = False,
        notice: Optional[Any] = None,
        move_sans: Optional[List[str]] = None,
        move_scroll: int = 0,
    ) -> None:
        s = self.s
        panel = pygame.Rect(self.panel_x, self.board_offset_y, self.panel_width, self.board_size)
        self._card(surface, panel, COLOR_PANEL)

        x = self.panel_x + s(16)
        y = self.board_offset_y + s(16)
        self._text(surface, self.font_lg, "Live Analysis", COLOR_TEXT, x, y)
        y += s(26)

        if notice is not None:
            sub = getattr(notice, "subtitle", "")
            turn_label = f"Result: {notice.title} ({sub})" if sub else f"Result: {notice.title}"
            tone = getattr(notice, "tone", "draw")
            turn_color = COLOR_PRIMARY if tone == "win" else (COLOR_DANGER if tone == "loss" else COLOR_TEXT)
        elif is_thinking:
            turn_label = "Turn: Agent thinking..."
            turn_color = COLOR_PRIMARY
        else:
            turn_label = "Turn: White to move" if board.turn == chess.WHITE else "Turn: Black to move"
            turn_color = COLOR_TEXT_MUTED
        self._text(surface, self.font_md, turn_label, turn_color, x, y)
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
        card = pygame.Rect(x, y, bar_w, s(144))
        pygame.draw.rect(surface, COLOR_PANEL_CARD, card, border_radius=s(8))
        tx = self.panel_x + s(24)
        self._text(surface, self.font_md, "Top Candidate Moves (MCTS)", COLOR_TEXT, tx, y + s(8))
        self._text(surface, self.font_sm, "Move        Visits     Share        Q", COLOR_TEXT_MUTED, tx, y + s(28))

        row_y = y + s(48)
        if analysis and analysis.candidates:
            for idx, c in enumerate(analysis.candidates[:4]):
                row = f"#{idx + 1}  {c.san:<8} {c.visits:<8} {c.visit_share * 100:>5.1f}%   {c.q_value:>+5.2f}"
                self._text(surface, self.font_sm, row, COLOR_PRIMARY if idx == 0 else COLOR_TEXT, tx, row_y)
                row_y += s(22)
        else:
            self._text(surface, self.font_sm, "Awaiting search...", COLOR_TEXT_MUTED, tx, row_y)

        # Move List & Material Difference card
        moves_card = self.play_moves_card_rect()
        pygame.draw.rect(surface, COLOR_PANEL_CARD, moves_card, border_radius=s(6))
        pygame.draw.rect(surface, COLOR_BORDER, moves_card, width=1, border_radius=s(6))

        # Header: Title on left, Material Diff on right
        hdr_txt = self.font_md.render("Move List", True, COLOR_TEXT)
        surface.blit(hdr_txt, (moves_card.x + s(10), moves_card.y + s(7)))

        balance = material_balance_white(board)
        mat_lbl = format_material_balance(board)
        if balance > 0:
            mat_color = (235, 235, 245)
        elif balance < 0:
            mat_color = (160, 160, 175)
        else:
            mat_color = COLOR_TEXT_MUTED
        txt_mat = self.font_sm.render(mat_lbl, True, mat_color)
        surface.blit(txt_mat, (moves_card.right - txt_mat.get_width() - s(10), moves_card.y + s(8)))

        # Header divider
        pygame.draw.line(surface, COLOR_BORDER, (moves_card.x, moves_card.y + s(28)), (moves_card.right, moves_card.y + s(28)))

        # Move list items
        moves_clip = pygame.Rect(moves_card.x + s(6), moves_card.y + s(32), moves_card.width - s(12), moves_card.height - s(36))
        sans = move_sans or []
        total_rows = (len(sans) + 1) // 2

        if not sans:
            no_moves = self.font_sm.render("No moves played yet", True, COLOR_TEXT_MUTED)
            surface.blit(no_moves, (moves_clip.x + s(10), moves_clip.y + s(8)))
        else:
            row_h = s(20)
            visible_rows = max(1, moves_clip.height // row_h)
            max_scroll = max(0, total_rows - visible_rows)
            scroll = max(0, min(max_scroll, move_scroll))

            orig_clip = surface.get_clip()
            surface.set_clip(moves_clip)

            num_col_w = s(32)
            half_gap = s(6)
            sb_pad = s(8) if total_rows > visible_rows else 0
            half_w = max(s(48), (moves_clip.width - num_col_w - half_gap - sb_pad) // 2)
            white_x = moves_clip.x + num_col_w
            black_x = white_x + half_w + half_gap

            for row_i in range(total_rows):
                row_y = moves_clip.y + (row_i - scroll) * row_h
                if row_y + row_h < moves_clip.y or row_y > moves_clip.bottom:
                    continue

                # Move number: " 1."
                num_surf = self.font_mono.render(f"{row_i + 1:>3}.", True, COLOR_TEXT_MUTED)
                surface.blit(num_surf, (moves_clip.x + s(2), row_y + s(2)))

                # White move
                w_idx = row_i * 2
                is_active_w = (w_idx == len(sans) - 1)
                cell_w = pygame.Rect(white_x, row_y, half_w, row_h)
                if is_active_w:
                    pygame.draw.rect(surface, (36, 60, 42), cell_w, border_radius=s(3))
                col_w = COLOR_PRIMARY if is_active_w else COLOR_TEXT
                surf_w = self.font_mono.render(sans[w_idx], True, col_w)
                surface.blit(surf_w, (cell_w.x + s(4), row_y + s(2)))

                # Black move
                b_idx = w_idx + 1
                if b_idx < len(sans):
                    is_active_b = (b_idx == len(sans) - 1)
                    cell_b = pygame.Rect(black_x, row_y, half_w, row_h)
                    if is_active_b:
                        pygame.draw.rect(surface, (36, 60, 42), cell_b, border_radius=s(3))
                    col_b = COLOR_PRIMARY if is_active_b else COLOR_TEXT
                    surf_b = self.font_mono.render(sans[b_idx], True, col_b)
                    surface.blit(surf_b, (cell_b.x + s(4), row_y + s(2)))

            surface.set_clip(orig_clip)

            # Scrollbar indicator
            if total_rows > visible_rows:
                sb_w = s(4)
                sb_x = moves_card.right - s(8)
                sb_y = moves_clip.y + s(2)
                sb_h = moves_clip.height - s(4)
                pygame.draw.rect(surface, (40, 43, 48), (sb_x, sb_y, sb_w, sb_h), border_radius=s(2))
                thumb_h = max(s(14), int(sb_h * (visible_rows / total_rows)))
                thumb_y = sb_y + int((sb_h - thumb_h) * (scroll / max_scroll)) if max_scroll > 0 else sb_y
                pygame.draw.rect(surface, (90, 95, 105), (sb_x, thumb_y, sb_w, thumb_h), border_radius=s(2))

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
