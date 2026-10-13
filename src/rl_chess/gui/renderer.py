"""Board, pieces, and live analysis rendering for Pygame."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import chess
import pygame

from rl_chess.mcts.search import MCTSAnalysis

# Colors
COLOR_LIGHT_SQ = (240, 217, 181)
COLOR_DARK_SQ = (181, 136, 99)
COLOR_SELECTED = (205, 210, 106, 180)
COLOR_LAST_MOVE = (205, 210, 106, 120)
COLOR_LEGAL_DOT = (100, 110, 64, 180)
COLOR_BG_PANEL = (32, 34, 37)
COLOR_PANEL_CARD = (47, 49, 54)
COLOR_TEXT_PRIMARY = (240, 240, 240)
COLOR_TEXT_MUTED = (160, 165, 175)
COLOR_TEXT_ACCENT = (76, 175, 80)
COLOR_BAR_WHITE = (235, 235, 235)
COLOR_BAR_BLACK = (50, 50, 50)
COLOR_ARROW_TOP1 = (52, 168, 83, 190)
COLOR_ARROW_TOP2 = (66, 133, 244, 160)

# Unicode Chess Characters
PIECE_UNICODE = {
    chess.Piece(chess.KING, chess.WHITE): "\u2654",
    chess.Piece(chess.QUEEN, chess.WHITE): "\u2655",
    chess.Piece(chess.ROOK, chess.WHITE): "\u2656",
    chess.Piece(chess.BISHOP, chess.WHITE): "\u2657",
    chess.Piece(chess.KNIGHT, chess.WHITE): "\u2658",
    chess.Piece(chess.PAWN, chess.WHITE): "\u2659",
    chess.Piece(chess.KING, chess.BLACK): "\u265a",
    chess.Piece(chess.QUEEN, chess.BLACK): "\u265b",
    chess.Piece(chess.ROOK, chess.BLACK): "\u265c",
    chess.Piece(chess.BISHOP, chess.BLACK): "\u265d",
    chess.Piece(chess.KNIGHT, chess.BLACK): "\u265e",
    chess.Piece(chess.PAWN, chess.BLACK): "\u265f",
}


class ChessRenderer:
    def __init__(
        self,
        board_size: int = 560,
        panel_width: int = 360,
    ):
        self.board_size = board_size
        self.panel_width = panel_width
        self.sq_size = board_size // 8

        # Initialize fonts
        pygame.font.init()
        font_name = "arialunicode" if "arialunicode" in pygame.font.get_fonts() else "Arial"
        self.piece_font = pygame.font.SysFont(font_name, int(self.sq_size * 0.75))
        self.font_lg = pygame.font.SysFont("Helvetica", 20, bold=True)
        self.font_md = pygame.font.SysFont("Helvetica", 14)
        self.font_sm = pygame.font.SysFont("Helvetica", 12)

    def square_to_rect(self, sq: chess.Square, flipped: bool = False) -> pygame.Rect:
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)

        col = file if not flipped else 7 - file
        row = 7 - rank if not flipped else rank

        return pygame.Rect(col * self.sq_size, row * self.sq_size, self.sq_size, self.sq_size)

    def pos_to_square(self, pos: Tuple[int, int], flipped: bool = False) -> Optional[chess.Square]:
        x, y = pos
        if x >= self.board_size or y >= self.board_size:
            return None
        col = x // self.sq_size
        row = y // self.sq_size

        file = col if not flipped else 7 - col
        rank = 7 - row if not flipped else row
        return chess.square(file, rank)

    def draw_board(
        self,
        surface: pygame.Surface,
        board: chess.Board,
        flipped: bool = False,
        selected_sq: Optional[chess.Square] = None,
        last_move: Optional[chess.Move] = None,
        legal_destinations: Optional[List[chess.Square]] = None,
        analysis: Optional[MCTSAnalysis] = None,
    ) -> None:
        """Draw board squares, coordinates, highlights, and pieces."""
        # 1. Squares
        for r in range(8):
            for c in range(8):
                color = COLOR_LIGHT_SQ if (r + c) % 2 == 0 else COLOR_DARK_SQ
                rect = pygame.Rect(c * self.sq_size, r * self.sq_size, self.sq_size, self.sq_size)
                pygame.draw.rect(surface, color, rect)

        # 2. Last move highlight
        if last_move is not None:
            surf_hl = pygame.Surface((self.sq_size, self.sq_size), pygame.SRCALPHA)
            surf_hl.fill(COLOR_LAST_MOVE)
            surface.blit(surf_hl, self.square_to_rect(last_move.from_square, flipped))
            surface.blit(surf_hl, self.square_to_rect(last_move.to_square, flipped))

        # 3. Selected square highlight
        if selected_sq is not None:
            surf_sel = pygame.Surface((self.sq_size, self.sq_size), pygame.SRCALPHA)
            surf_sel.fill(COLOR_SELECTED)
            surface.blit(surf_sel, self.square_to_rect(selected_sq, flipped))

        # 4. Legal destination markers
        if legal_destinations:
            for dest in legal_destinations:
                rect = self.square_to_rect(dest, flipped)
                dot_center = rect.center
                piece_at_dest = board.piece_at(dest)
                if piece_at_dest:
                    # Ring for captures
                    pygame.draw.circle(surface, (100, 110, 64), dot_center, int(self.sq_size * 0.42), 4)
                else:
                    # Small dot for moves
                    pygame.draw.circle(surface, (100, 110, 64), dot_center, int(self.sq_size * 0.16))

        # 5. Top candidate move arrows from analysis
        if analysis and analysis.candidates:
            self._draw_candidate_arrows(surface, analysis.candidates[:2], flipped)

        # 6. Pieces
        for sq, piece in board.piece_map().items():
            rect = self.square_to_rect(sq, flipped)
            char = PIECE_UNICODE.get(piece, "")
            # Shadow
            shadow_surf = self.piece_font.render(char, True, (30, 30, 30))
            shadow_rect = shadow_surf.get_rect(center=(rect.centerx + 1, rect.centery + 1))
            surface.blit(shadow_surf, shadow_rect)
            # Main piece glyph
            p_color = (250, 250, 250) if piece.color == chess.WHITE else (20, 20, 20)
            text_surf = self.piece_font.render(char, True, p_color)
            text_rect = text_surf.get_rect(center=rect.center)
            surface.blit(text_surf, text_rect)

        # 7. Coordinates
        for i in range(8):
            rank_label = str(8 - i if not flipped else i + 1)
            file_label = chr(ord('a') + (i if not flipped else 7 - i))
            c_color = COLOR_DARK_SQ if i % 2 == 0 else COLOR_LIGHT_SQ

            r_txt = self.font_sm.render(rank_label, True, c_color)
            surface.blit(r_txt, (4, i * self.sq_size + 4))

            f_txt = self.font_sm.render(file_label, True, c_color)
            surface.blit(f_txt, ((i + 1) * self.sq_size - 12, self.board_size - 16))

    def _draw_candidate_arrows(
        self,
        surface: pygame.Surface,
        candidates: List,
        flipped: bool,
    ) -> None:
        colors = [COLOR_ARROW_TOP1, COLOR_ARROW_TOP2]
        for idx, cand in enumerate(candidates):
            color = colors[idx % len(colors)]
            start_rect = self.square_to_rect(cand.move.from_square, flipped)
            end_rect = self.square_to_rect(cand.move.to_square, flipped)

            start = (start_rect.centerx, start_rect.centery)
            end = (end_rect.centerx, end_rect.centery)

            # Draw line
            pygame.draw.line(surface, color[:3], start, end, width=4)
            # Draw arrowhead
            angle = math.atan2(end[1] - start[1], end[0] - start[0])
            arrow_len = 12
            arrow_w = math.pi / 6
            p1 = (end[0] - arrow_len * math.cos(angle - arrow_w), end[1] - arrow_len * math.sin(angle - arrow_w))
            p2 = (end[0] - arrow_len * math.cos(angle + arrow_w), end[1] - arrow_len * math.sin(angle + arrow_w))
            pygame.draw.polygon(surface, color[:3], [end, p1, p2])

    def draw_analysis_panel(
        self,
        surface: pygame.Surface,
        board: chess.Board,
        analysis: Optional[MCTSAnalysis],
        agent_name: str,
        is_thinking: bool,
    ) -> None:
        """Render side panel with evaluation bar, candidate moves, and PV."""
        panel_x = self.board_size
        panel_rect = pygame.Rect(panel_x, 0, self.panel_width, self.board_size)
        pygame.draw.rect(surface, COLOR_BG_PANEL, panel_rect)

        y = 16
        # Header
        title = self.font_lg.render("Live Analysis", True, COLOR_TEXT_PRIMARY)
        surface.blit(title, (panel_x + 16, y))
        y += 30

        status_text = "Agent thinking..." if is_thinking else ("White to move" if board.turn == chess.WHITE else "Black to move")
        color_status = COLOR_TEXT_ACCENT if is_thinking else COLOR_TEXT_MUTED
        stat = self.font_md.render(f"Turn: {status_text}", True, color_status)
        surface.blit(stat, (panel_x + 16, y))
        y += 24

        agent_label = self.font_sm.render(f"Agent: {agent_name}", True, COLOR_TEXT_MUTED)
        surface.blit(agent_label, (panel_x + 16, y))
        y += 26

        # Evaluation Bar (Win Probability)
        root_val = analysis.root_value if analysis else 0.0
        # If Black to move, root_val is from Black perspective; invert to get White-advantage scale
        white_val = root_val if board.turn == chess.WHITE else -root_val
        # Map [-1, 1] to [0%, 100%] White win expectancy
        white_pct = (white_val + 1.0) / 2.0

        bar_label = self.font_md.render(f"Win Expectancy: {white_pct * 100:.1f}% White", True, COLOR_TEXT_PRIMARY)
        surface.blit(bar_label, (panel_x + 16, y))
        y += 20

        # Bar track
        bar_w = self.panel_width - 32
        bar_h = 16
        pygame.draw.rect(surface, COLOR_BAR_BLACK, (panel_x + 16, y, bar_w, bar_h), border_radius=4)
        white_w = int(bar_w * white_pct)
        if white_w > 0:
            pygame.draw.rect(surface, COLOR_BAR_WHITE, (panel_x + 16, y, white_w, bar_h), border_radius=4)
        y += 28

        # Candidate Moves Section
        card_rect = pygame.Rect(panel_x + 16, y, self.panel_width - 32, 170)
        pygame.draw.rect(surface, COLOR_PANEL_CARD, card_rect, border_radius=8)

        card_title = self.font_md.render("Top Candidate Moves (MCTS)", True, COLOR_TEXT_PRIMARY)
        surface.blit(card_title, (panel_x + 24, y + 10))

        headers = self.font_sm.render("Move        Visits     Share        Q", True, COLOR_TEXT_MUTED)
        surface.blit(headers, (panel_x + 24, y + 32))

        cand_y = y + 50
        if analysis and analysis.candidates:
            for idx, c in enumerate(analysis.candidates[:4]):
                row_txt = f"#{idx + 1}  {c.san:<8} {c.visits:<8} {c.visit_share * 100:>5.1f}%   {c.q_value:>+5.2f}"
                col = COLOR_TEXT_ACCENT if idx == 0 else COLOR_TEXT_PRIMARY
                surf_row = self.font_sm.render(row_txt, True, col)
                surface.blit(surf_row, (panel_x + 24, cand_y))
                cand_y += 24
        else:
            empty_txt = self.font_sm.render("Awaiting search...", True, COLOR_TEXT_MUTED)
            surface.blit(empty_txt, (panel_x + 24, cand_y))

        y += 182

        # Principal Variation (PV)
        pv_card = pygame.Rect(panel_x + 16, y, self.panel_width - 32, 60)
        pygame.draw.rect(surface, COLOR_PANEL_CARD, pv_card, border_radius=8)
        pv_title = self.font_sm.render("Principal Variation:", True, COLOR_TEXT_MUTED)
        surface.blit(pv_title, (panel_x + 24, y + 8))

        pv_text = analysis.pv_san if (analysis and analysis.pv_san) else "-"
        if len(pv_text) > 36:
            pv_text = pv_text[:33] + "..."
        pv_surf = self.font_md.render(pv_text, True, COLOR_TEXT_PRIMARY)
        surface.blit(pv_surf, (panel_x + 24, y + 28))

        y += 74
        # Controls info
        keys_info = [
            "Hotkeys:",
            "  [N] New Game    [F] Flip Board",
            "  [U] Undo Move   [Esc] Quit",
        ]
        for line in keys_info:
            txt = self.font_sm.render(line, True, COLOR_TEXT_MUTED)
            surface.blit(txt, (panel_x + 16, y))
            y += 18
