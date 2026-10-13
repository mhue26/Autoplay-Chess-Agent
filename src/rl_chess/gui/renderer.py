"""Board, pieces, analysis, and multi-tab dashboard rendering for Pygame."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import chess
import pygame

from rl_chess.gui.widgets import (
    COLOR_BG,
    COLOR_BORDER,
    COLOR_CARD,
    COLOR_PRIMARY,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
)
from rl_chess.mcts.search import MCTSAnalysis

# Board Colors
COLOR_LIGHT_SQ = (240, 217, 181)
COLOR_DARK_SQ = (181, 136, 99)
COLOR_SELECTED = (205, 210, 106, 180)
COLOR_LAST_MOVE = (205, 210, 106, 120)
COLOR_LEGAL_DOT = (100, 110, 64, 180)
COLOR_BG_PANEL = (32, 34, 37)
COLOR_PANEL_CARD = (47, 49, 54)
COLOR_TEXT_PRIMARY = (240, 240, 240)
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
        board_offset_x: int = 20,
        board_offset_y: int = 50,
        panel_x: int = 600,
        panel_width: int = 420,
    ):
        self.board_size = board_size
        self.board_offset_x = board_offset_x
        self.board_offset_y = board_offset_y
        self.panel_x = panel_x
        self.panel_width = panel_width
        self.sq_size = board_size // 8

        # Initialize fonts
        pygame.font.init()
        font_name = "arialunicode" if "arialunicode" in pygame.font.get_fonts() else "Arial"
        self.piece_font = pygame.font.SysFont(font_name, int(self.sq_size * 0.75))
        self.font_title = pygame.font.SysFont("Helvetica", 22, bold=True)
        self.font_lg = pygame.font.SysFont("Helvetica", 17, bold=True)
        self.font_md = pygame.font.SysFont("Helvetica", 14)
        self.font_sm = pygame.font.SysFont("Helvetica", 12)
        self.font_mono = pygame.font.SysFont("Courier", 13)

    def square_to_rect(self, sq: chess.Square, flipped: bool = False) -> pygame.Rect:
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)

        col = file if not flipped else 7 - file
        row = 7 - rank if not flipped else rank

        return pygame.Rect(
            self.board_offset_x + col * self.sq_size,
            self.board_offset_y + row * self.sq_size,
            self.sq_size,
            self.sq_size,
        )

    def pos_to_square(self, pos: Tuple[int, int], flipped: bool = False) -> Optional[chess.Square]:
        x, y = pos
        rel_x = x - self.board_offset_x
        rel_y = y - self.board_offset_y
        if rel_x < 0 or rel_y < 0 or rel_x >= self.board_size or rel_y >= self.board_size:
            return None

        col = rel_x // self.sq_size
        row = rel_y // self.sq_size

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
                rect = pygame.Rect(
                    self.board_offset_x + c * self.sq_size,
                    self.board_offset_y + r * self.sq_size,
                    self.sq_size,
                    self.sq_size,
                )
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
                    pygame.draw.circle(surface, (100, 110, 64), dot_center, int(self.sq_size * 0.42), 4)
                else:
                    pygame.draw.circle(surface, (100, 110, 64), dot_center, int(self.sq_size * 0.16))

        # 5. Top candidate move arrows from analysis
        if analysis and analysis.candidates:
            self._draw_candidate_arrows(surface, analysis.candidates[:2], flipped)

        # 6. Pieces
        for sq, piece in board.piece_map().items():
            rect = self.square_to_rect(sq, flipped)
            char = PIECE_UNICODE.get(piece, "")
            shadow_surf = self.piece_font.render(char, True, (30, 30, 30))
            shadow_rect = shadow_surf.get_rect(center=(rect.centerx + 1, rect.centery + 1))
            surface.blit(shadow_surf, shadow_rect)
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
            surface.blit(r_txt, (self.board_offset_x + 4, self.board_offset_y + i * self.sq_size + 4))

            f_txt = self.font_sm.render(file_label, True, c_color)
            surface.blit(
                f_txt,
                (self.board_offset_x + (i + 1) * self.sq_size - 12, self.board_offset_y + self.board_size - 16),
            )

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

            pygame.draw.line(surface, color[:3], start, end, width=4)
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
        panel_rect = pygame.Rect(self.panel_x, self.board_offset_y, self.panel_width, self.board_size)
        pygame.draw.rect(surface, COLOR_BG_PANEL, panel_rect, border_radius=8)
        pygame.draw.rect(surface, COLOR_BORDER, panel_rect, width=1, border_radius=8)

        y = self.board_offset_y + 16
        title = self.font_lg.render("Live Analysis", True, COLOR_TEXT_PRIMARY)
        surface.blit(title, (self.panel_x + 16, y))
        y += 26

        status_text = "Agent thinking..." if is_thinking else ("White to move" if board.turn == chess.WHITE else "Black to move")
        color_status = COLOR_TEXT_ACCENT if is_thinking else COLOR_TEXT_MUTED
        stat = self.font_md.render(f"Turn: {status_text}", True, color_status)
        surface.blit(stat, (self.panel_x + 16, y))
        y += 22

        agent_label = self.font_sm.render(f"Agent: {agent_name}", True, COLOR_TEXT_MUTED)
        surface.blit(agent_label, (self.panel_x + 16, y))
        y += 24

        # Evaluation Bar
        root_val = analysis.root_value if analysis else 0.0
        white_val = root_val if board.turn == chess.WHITE else -root_val
        white_pct = (white_val + 1.0) / 2.0

        bar_label = self.font_md.render(f"Win Expectancy: {white_pct * 100:.1f}% White", True, COLOR_TEXT_PRIMARY)
        surface.blit(bar_label, (self.panel_x + 16, y))
        y += 20

        bar_w = self.panel_width - 32
        bar_h = 16
        pygame.draw.rect(surface, COLOR_BAR_BLACK, (self.panel_x + 16, y, bar_w, bar_h), border_radius=4)
        white_w = int(bar_w * white_pct)
        if white_w > 0:
            pygame.draw.rect(surface, COLOR_BAR_WHITE, (self.panel_x + 16, y, white_w, bar_h), border_radius=4)
        y += 26

        # Candidate Moves Section
        card_rect = pygame.Rect(self.panel_x + 16, y, self.panel_width - 32, 175)
        pygame.draw.rect(surface, COLOR_PANEL_CARD, card_rect, border_radius=8)

        card_title = self.font_md.render("Top Candidate Moves (MCTS)", True, COLOR_TEXT_PRIMARY)
        surface.blit(card_title, (self.panel_x + 24, y + 10))

        headers = self.font_sm.render("Move        Visits     Share        Q", True, COLOR_TEXT_MUTED)
        surface.blit(headers, (self.panel_x + 24, y + 32))

        cand_y = y + 52
        if analysis and analysis.candidates:
            for idx, c in enumerate(analysis.candidates[:4]):
                row_txt = f"#{idx + 1}  {c.san:<8} {c.visits:<8} {c.visit_share * 100:>5.1f}%   {c.q_value:>+5.2f}"
                col = COLOR_TEXT_ACCENT if idx == 0 else COLOR_TEXT_PRIMARY
                surf_row = self.font_sm.render(row_txt, True, col)
                surface.blit(surf_row, (self.panel_x + 24, cand_y))
                cand_y += 24
        else:
            empty_txt = self.font_sm.render("Awaiting search...", True, COLOR_TEXT_MUTED)
            surface.blit(empty_txt, (self.panel_x + 24, cand_y))

        y += 186

        # Principal Variation
        pv_card = pygame.Rect(self.panel_x + 16, y, self.panel_width - 32, 60)
        pygame.draw.rect(surface, COLOR_PANEL_CARD, pv_card, border_radius=8)
        pv_title = self.font_sm.render("Principal Variation:", True, COLOR_TEXT_MUTED)
        surface.blit(pv_title, (self.panel_x + 24, y + 8))

        pv_text = analysis.pv_san if (analysis and analysis.pv_san) else "-"
        if len(pv_text) > 42:
            pv_text = pv_text[:39] + "..."
        pv_surf = self.font_md.render(pv_text, True, COLOR_TEXT_PRIMARY)
        surface.blit(pv_surf, (self.panel_x + 24, y + 28))

        y += 76
        keys_info = [
            "Hotkeys:",
            "  [N] New Game    [F] Flip Board",
            "  [U] Undo Move   [Esc] Quit",
        ]
        for line in keys_info:
            txt = self.font_sm.render(line, True, COLOR_TEXT_MUTED)
            surface.blit(txt, (self.panel_x + 16, y))
            y += 18

    # -------------------------------------------------------------
    # Multi-tab Dashboard Views
    # -------------------------------------------------------------

    def draw_train_tab_header(
        self,
        surface: pygame.Surface,
        status: str,
        champion_name: str,
    ) -> None:
        """Render controls and headers for the training tab."""
        x = 20
        y = 54
        title = self.font_title.render("Self-Play Reinforcement Learning", True, COLOR_TEXT)
        surface.blit(title, (x, y))

        status_c = COLOR_TEXT_ACCENT if status == "running" else COLOR_TEXT_MUTED
        stat_surf = self.font_md.render(f"Status: {status.upper()}", True, status_c)
        surface.blit(stat_surf, (x + 450, y + 6))

        champ_surf = self.font_md.render(f"Current Champion: {champion_name}", True, COLOR_TEXT)
        surface.blit(champ_surf, (x + 650, y + 6))

    def draw_benchmark_tab_header(
        self,
        surface: pygame.Surface,
        status: str,
        checkpoint_name: str,
        latest_elo: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Render controls and headers for the benchmark tab."""
        x = 20
        y = 54
        title = self.font_title.render("Stockfish Gauntlet & Calibrated Elo", True, COLOR_TEXT)
        surface.blit(title, (x, y))

        status_c = COLOR_TEXT_ACCENT if status == "running" else COLOR_TEXT_MUTED
        stat_surf = self.font_md.render(f"Status: {status.upper()}", True, status_c)
        surface.blit(stat_surf, (x + 480, y + 6))

        # Elo badge if available
        if latest_elo:
            badge_rect = pygame.Rect(x + 650, y - 2, 330, 36)
            pygame.draw.rect(surface, COLOR_PANEL_CARD, badge_rect, border_radius=6)
            pygame.draw.rect(surface, COLOR_PRIMARY, badge_rect, width=1, border_radius=6)

            txt = f"Latest Elo: {latest_elo.get('elo', 0):.0f} (CI: [{latest_elo.get('ci_low', 0):.0f}, {latest_elo.get('ci_high', 0):.0f}])"
            b_surf = self.font_md.render(txt, True, COLOR_PRIMARY)
            surface.blit(b_surf, (badge_rect.x + 12, badge_rect.y + 9))

    def draw_stats_tab(
        self,
        surface: pygame.Surface,
        summary: Dict[str, Any],
        elo_history: List[Dict[str, Any]],
        checkpoints: List[Any],
    ) -> None:
        """Render universal training metrics cards and historical tables."""
        x = 20
        y = 54

        title = self.font_title.render("Training & Evaluation Ledger", True, COLOR_TEXT)
        surface.blit(title, (x, y))
        y += 40

        # 4 Summary Cards
        card_w = 235
        card_h = 75
        metrics = [
            ("Total Games", str(summary.get("total_games", 0)), f"Self-Play: {summary.get('selfplay_games', 0)}"),
            ("Training Steps", str(summary.get("training_steps", 0)), f"~{summary.get('selfplay_positions_estimated', 0)} positions"),
            ("Checkpoints", str(summary.get("checkpoints_count", 0)), f"Champ: {summary.get('champion', {}).get('name', 'None') if summary.get('champion') else 'None'}"),
            ("Latest Elo", f"{summary.get('latest_elo', {}).get('elo', 0):.0f}" if summary.get("latest_elo") else "N/A", "Stockfish Anchor"),
        ]

        for i, (label, val, sub) in enumerate(metrics):
            cx = x + i * (card_w + 20)
            rect = pygame.Rect(cx, y, card_w, card_h)
            pygame.draw.rect(surface, COLOR_PANEL_CARD, rect, border_radius=8)
            pygame.draw.rect(surface, COLOR_BORDER, rect, width=1, border_radius=8)

            lbl_s = self.font_sm.render(label, True, COLOR_TEXT_MUTED)
            val_s = self.font_title.render(val, True, COLOR_PRIMARY if i == 3 else COLOR_TEXT)
            sub_s = self.font_sm.render(sub, True, COLOR_TEXT_MUTED)

            surface.blit(lbl_s, (cx + 14, y + 8))
            surface.blit(val_s, (cx + 14, y + 26))
            surface.blit(sub_s, (cx + 14, y + 54))

        y += 95

        # Elo History Section
        h_rect = pygame.Rect(x, y, 620, 390)
        pygame.draw.rect(surface, COLOR_PANEL_CARD, h_rect, border_radius=8)
        pygame.draw.rect(surface, COLOR_BORDER, h_rect, width=1, border_radius=8)

        sec_title = self.font_lg.render("Calibrated Elo History", True, COLOR_TEXT)
        surface.blit(sec_title, (x + 16, y + 12))

        table_header = self.font_mono.render(
            f"{'Checkpoint':<14} {'Gen':<5} {'Elo':<8} {'95% CI':<14} {'Games':<6} {'Date':<10}",
            True,
            COLOR_TEXT_MUTED,
        )
        surface.blit(table_header, (x + 16, y + 42))
        pygame.draw.line(surface, COLOR_BORDER, (x + 16, y + 62), (x + 604, y + 62))

        row_y = y + 70
        if elo_history:
            for row in elo_history[-11:]:  # show latest 11
                ci = f"[{row.get('ci_low', 0):.0f}, {row.get('ci_high', 0):.0f}]"
                dt = str(row.get("created_at", ""))[:10]
                line_str = f"{row.get('checkpoint_name', ''):<14} {row.get('generation', 0):<5} {row.get('elo', 0):<8.0f} {ci:<14} {row.get('n_games', 0):<6} {dt:<10}"
                surf_line = self.font_mono.render(line_str, True, COLOR_TEXT)
                surface.blit(surf_line, (x + 16, row_y))
                row_y += 26
        else:
            none_s = self.font_md.render("No Elo estimates recorded yet. Run a benchmark!", True, COLOR_TEXT_MUTED)
            surface.blit(none_s, (x + 16, row_y))

        # Checkpoints Section
        cp_rect = pygame.Rect(x + 640, y, 360, 390)
        pygame.draw.rect(surface, COLOR_PANEL_CARD, cp_rect, border_radius=8)
        pygame.draw.rect(surface, COLOR_BORDER, cp_rect, width=1, border_radius=8)

        cp_title = self.font_lg.render("Registered Checkpoints", True, COLOR_TEXT)
        surface.blit(cp_title, (x + 656, y + 12))

        cp_header = self.font_mono.render(f"{'Name':<12} {'Gen':<5} {'Promoted':<10}", True, COLOR_TEXT_MUTED)
        surface.blit(cp_header, (x + 656, y + 42))
        pygame.draw.line(surface, COLOR_BORDER, (x + 656, y + 62), (x + 984, y + 62))

        cp_row_y = y + 70
        if checkpoints:
            for cp in checkpoints[-11:]:
                prom = "Yes (Champ)" if getattr(cp, "promoted", False) else "No"
                cp_str = f"{getattr(cp, 'name', ''):<12} {getattr(cp, 'generation', 0):<5} {prom:<10}"
                col = COLOR_PRIMARY if getattr(cp, "promoted", False) else COLOR_TEXT
                surf_cp = self.font_mono.render(cp_str, True, col)
                surface.blit(surf_cp, (x + 656, cp_row_y))
                cp_row_y += 26
        else:
            none_cp = self.font_md.render("No checkpoints saved yet.", True, COLOR_TEXT_MUTED)
            surface.blit(none_cp, (x + 656, cp_row_y))
