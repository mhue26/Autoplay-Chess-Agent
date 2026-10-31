"""Training Visualiser: state reducer, MCTS telemetry, and live Pygame visualizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import chess
import pygame

from rl_chess.game import format_material_balance, material_balance_white
from rl_chess.gui.renderer import ChessRenderer
from rl_chess.gui.widgets import (
    COLOR_BORDER,
    COLOR_PANEL_CARD,
    COLOR_PRIMARY,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
)

COLOR_PRIMARY_DARK = (46, 125, 50)
from rl_chess.mcts import CandidateMoveAnalysis, MCTSAnalysis


@dataclass
class CandidateMoveInfo:
    san: str
    uci: str
    from_sq: int
    to_sq: int
    prior: float
    visits: int
    visit_share: float
    q_value: float


@dataclass
class MoveRecord:
    ply: int
    san: str
    uci: str
    from_sq: int
    to_sq: int
    mover: str  # "white" or "black"
    fen: str
    temperature: float
    add_noise: bool
    sims: int
    root_value: float
    root_value_white: float
    candidates: List[CandidateMoveInfo]
    pv: List[str]
    pv_san: str
    entropy: float


@dataclass
class GameRecord:
    phase: str  # "selfplay" or "arena"
    generation: int
    game_idx: int
    total_games: int
    white: str
    black: str
    moves: List[MoveRecord] = field(default_factory=list)
    result: Optional[str] = None
    plies: int = 0
    termination_reason: str = ""
    white_val: float = 0.0
    samples_count: int = 0
    is_finished: bool = False
    final_fen: str = ""
    candidate_color: Optional[str] = None  # arena: which side the candidate (updated agent) plays


@dataclass
class TrainStepRecord:
    generation: int
    step: int
    total_steps: int
    loss_total: float
    loss_policy: float
    loss_value: float
    lr: float
    replay_buffer_size: int


@dataclass
class ArenaGameRecord:
    generation: int
    game_idx: int
    total_games: int
    candidate_color: str
    result: str
    candidate_points: float
    plies: int
    running_score: float
    threshold: float


PACE_OPTIONS = ["Max", "0.2s", "0.5s", "1.0s"]
PACE_DELAYS = {"Max": 0.0, "0.2s": 0.2, "0.5s": 0.5, "1.0s": 1.0}


class TrainingVizState:
    """State reducer for streaming training telemetry events."""

    def __init__(self, max_finished_games: int = 50):
        self.max_finished_games = max_finished_games

        # Generation & Phase state
        self.phase: str = "idle"  # idle, selfplay, training, arena, completed, stopped
        self.generation: int = 0
        self.gen_index: int = 0
        self.total_generations: int = 0
        self.champion_name: str = "None"
        self.champion_gen: int = 0
        self.games_per_gen: int = 10
        self.sims: int = 64
        self.temp_init: float = 1.0
        self.temp_final: float = 0.0
        self.temp_decay_plies: int = 30
        self.replay_buffer_size: int = 0

        # Games
        self.current_game: Optional[GameRecord] = None
        self.finished_games: List[GameRecord] = []

        # Training loss history
        self.loss_history: List[TrainStepRecord] = []

        # Arena gating records
        self.arena_games: List[ArenaGameRecord] = []
        self.arena_total_games: int = 10
        self.arena_threshold: float = 0.55
        self.arena_candidate_wins: int = 0
        self.arena_candidate_draws: int = 0
        self.arena_candidate_losses: int = 0

        # Summary of last completed generation
        self.last_gen_summary: Optional[Dict[str, Any]] = None

        # Interactive inspection / scrubbing
        self.viewing_finished_idx: Optional[int] = None  # None -> viewing current_game
        self.viewing_ply: Optional[int] = None  # None -> viewing latest move
        self.follow_live: bool = True
        self.pace_mode: str = "Max"

    @property
    def pace_delay_sec(self) -> float:
        return PACE_DELAYS.get(self.pace_mode, 0.0)

    def cycle_pace(self) -> str:
        idx = (PACE_OPTIONS.index(self.pace_mode) + 1) % len(PACE_OPTIONS)
        self.pace_mode = PACE_OPTIONS[idx]
        return self.pace_mode

    def _default_total_games(self, phase: str) -> int:
        if phase == "arena":
            return self.arena_total_games
        return self.games_per_gen

    def _record_arena_game_result(self, event: Dict[str, Any]) -> None:
        pts = float(event.get("candidate_points", 0.5))
        if pts == 1.0:
            self.arena_candidate_wins += 1
        elif pts == 0.5:
            self.arena_candidate_draws += 1
        else:
            self.arena_candidate_losses += 1

        arena_rec = ArenaGameRecord(
            generation=int(event.get("generation", self.generation)),
            game_idx=int(event.get("game_idx", len(self.arena_games) + 1)),
            total_games=int(event.get("total_games", self.arena_total_games)),
            candidate_color=str(event.get("candidate_color", "white")),
            result=str(event.get("result", "1/2-1/2")),
            candidate_points=pts,
            plies=int(event.get("plies", 0)),
            running_score=float(event.get("running_score", 0.0)),
            threshold=float(event.get("threshold", self.arena_threshold)),
        )
        self.arena_games.append(arena_rec)
        self.arena_threshold = arena_rec.threshold

    def handle_event(self, event: Dict[str, Any]) -> None:
        """Process incoming train_viz event."""
        ev_type = event.get("type", "")

        if ev_type == "gen_start":
            self.phase = "selfplay"
            self.generation = int(event.get("generation", self.generation + 1))
            self.gen_index = int(event.get("gen_index", 1))
            self.total_generations = int(event.get("total_generations", 1))
            self.champion_name = str(event.get("champion_name", self.champion_name))
            self.champion_gen = int(event.get("champion_gen", self.champion_gen))
            self.games_per_gen = int(event.get("games_per_gen", self.games_per_gen))
            self.sims = int(event.get("sims", self.sims))
            self.temp_init = float(event.get("temp_init", self.temp_init))
            self.temp_final = float(event.get("temp_final", self.temp_final))
            self.temp_decay_plies = int(event.get("temp_decay_plies", self.temp_decay_plies))
            self.replay_buffer_size = int(event.get("replay_buffer_size", self.replay_buffer_size))
            self.loss_history = []
            self.arena_games = []
            self.arena_total_games = int(event.get("arena_num_games", self.arena_total_games))
            self.arena_candidate_wins = 0
            self.arena_candidate_draws = 0
            self.arena_candidate_losses = 0

        elif ev_type == "arena_start":
            self.phase = "arena"
            self.arena_total_games = int(event.get("total_games", self.arena_total_games))
            self.arena_threshold = float(event.get("threshold", self.arena_threshold))
            self.arena_games = []
            self.arena_candidate_wins = 0
            self.arena_candidate_draws = 0
            self.arena_candidate_losses = 0

        elif ev_type == "game_start":
            phase = event.get("phase", "selfplay")
            self.phase = phase
            default_total = self._default_total_games(phase)
            total = int(event.get("total_games", event.get("total", default_total)))
            cand_color = event.get("candidate_color")
            self.current_game = GameRecord(
                phase=phase,
                generation=int(event.get("generation", self.generation)),
                game_idx=int(event.get("game_idx", 1)),
                total_games=total,
                white=str(event.get("white", "White")),
                black=str(event.get("black", "Black")),
                candidate_color=str(cand_color) if cand_color else None,
            )
            if self.follow_live:
                self.viewing_finished_idx = None
                self.viewing_ply = None

        elif ev_type == "move":
            if self.current_game is None:
                phase = event.get("phase", "selfplay")
                default_total = self._default_total_games(phase)
                cand_color = event.get("candidate_color")
                self.current_game = GameRecord(
                    phase=phase,
                    generation=int(event.get("generation", self.generation)),
                    game_idx=int(event.get("game_idx", 1)),
                    total_games=int(event.get("total_games", default_total)),
                    white=str(event.get("white", "AlphaZero")),
                    black=str(event.get("black", "AlphaZero")),
                    candidate_color=str(cand_color) if cand_color else None,
                )

            raw_cands = event.get("candidates", [])
            candidates = [
                CandidateMoveInfo(
                    san=str(c.get("san", "")),
                    uci=str(c.get("uci", "")),
                    from_sq=int(c.get("from_sq", 0)),
                    to_sq=int(c.get("to_sq", 0)),
                    prior=float(c.get("prior", 0.0)),
                    visits=int(c.get("visits", 0)),
                    visit_share=float(c.get("visit_share", 0.0)),
                    q_value=float(c.get("q_value", 0.0)),
                )
                for c in raw_cands
            ]

            move_rec = MoveRecord(
                ply=int(event.get("ply", len(self.current_game.moves) + 1)),
                san=str(event.get("san", "")),
                uci=str(event.get("uci", "")),
                from_sq=int(event.get("from_sq", 0)),
                to_sq=int(event.get("to_sq", 0)),
                mover=str(event.get("mover", "white")),
                fen=str(event.get("fen", chess.STARTING_FEN)),
                temperature=float(event.get("temperature", 0.0)),
                add_noise=bool(event.get("add_noise", False)),
                sims=int(event.get("sims", self.sims)),
                root_value=float(event.get("root_value", 0.0)),
                root_value_white=float(event.get("root_value_white", 0.0)),
                candidates=candidates,
                pv=[str(m) for m in event.get("pv", [])],
                pv_san=str(event.get("pv_san", "")),
                entropy=float(event.get("entropy", 0.0)),
            )
            self.current_game.moves.append(move_rec)

            if self.follow_live and self.viewing_finished_idx is None:
                self.viewing_ply = len(self.current_game.moves)

        elif ev_type == "game_end":
            if self.current_game is not None:
                self.current_game.result = str(event.get("result", "1/2-1/2"))
                self.current_game.plies = int(event.get("plies", len(self.current_game.moves)))
                self.current_game.termination_reason = str(event.get("termination_reason", ""))
                self.current_game.white_val = float(event.get("white_val", 0.0))
                self.current_game.samples_count = int(event.get("samples_count", len(self.current_game.moves)))
                self.current_game.is_finished = True
                self.current_game.final_fen = str(event.get("final_fen", ""))

                self.finished_games.append(self.current_game)
                if len(self.finished_games) > self.max_finished_games:
                    self.finished_games.pop(0)

            if "replay_buffer_size" in event:
                self.replay_buffer_size = int(event["replay_buffer_size"])

        elif ev_type == "train_step":
            self.phase = "training"
            step_rec = TrainStepRecord(
                generation=int(event.get("generation", self.generation)),
                step=int(event.get("step", 1)),
                total_steps=int(event.get("total_steps", 100)),
                loss_total=float(event.get("loss_total", 0.0)),
                loss_policy=float(event.get("loss_policy", 0.0)),
                loss_value=float(event.get("loss_value", 0.0)),
                lr=float(event.get("lr", 0.0)),
                replay_buffer_size=int(event.get("replay_buffer_size", self.replay_buffer_size)),
            )
            self.loss_history.append(step_rec)
            self.replay_buffer_size = step_rec.replay_buffer_size

        elif ev_type == "arena_game":
            self.phase = "arena"
            self._record_arena_game_result(event)

        elif ev_type == "gen_end":
            self.last_gen_summary = {
                "generation": int(event.get("generation", self.generation)),
                "promoted": bool(event.get("promoted", False)),
                "win_rate": float(event.get("win_rate", 0.0)),
                "threshold": float(event.get("threshold", self.arena_threshold)),
                "champion_name": str(event.get("champion_name", self.champion_name)),
            }
            if self.last_gen_summary["promoted"]:
                self.champion_name = self.last_gen_summary["champion_name"]

    # ------------------------------------------------------------------
    # Query methods for inspection
    # ------------------------------------------------------------------

    def get_viewing_game(self) -> Optional[GameRecord]:
        """Return the GameRecord currently selected for viewing."""
        if self.viewing_finished_idx is not None:
            if 0 <= self.viewing_finished_idx < len(self.finished_games):
                return self.finished_games[self.viewing_finished_idx]
        return self.current_game

    def get_viewing_ply(self) -> int:
        """Return 0-indexed ply number currently displayed."""
        game = self.get_viewing_game()
        if not game or not game.moves:
            return 0
        if self.viewing_ply is None:
            return len(game.moves)
        return max(0, min(len(game.moves), self.viewing_ply))

    def get_viewing_board_and_move(self) -> Tuple[chess.Board, Optional[MoveRecord]]:
        """Return the chess.Board for current view and the MoveRecord that led to it."""
        game = self.get_viewing_game()
        if not game or not game.moves:
            return chess.Board(), None

        ply = self.get_viewing_ply()
        if ply == 0:
            return chess.Board(), None

        idx = min(ply - 1, len(game.moves) - 1)
        move_rec = game.moves[idx]
        try:
            board = chess.Board(move_rec.fen)
        except Exception:
            board = chess.Board()
        return board, move_rec

    def step_ply(self, delta: int) -> None:
        """Scrub forward or backward in ply history."""
        game = self.get_viewing_game()
        if not game or not game.moves:
            return
        self.follow_live = False
        cur = self.get_viewing_ply()
        nxt = max(0, min(len(game.moves), cur + delta))
        self.viewing_ply = nxt

    def go_to_start(self) -> None:
        self.follow_live = False
        self.viewing_ply = 0

    def go_to_end(self) -> None:
        game = self.get_viewing_game()
        if game and game.moves:
            self.viewing_ply = len(game.moves)
            if self.viewing_finished_idx is None:
                self.follow_live = True

    def select_game(self, finished_idx: Optional[int]) -> None:
        """Select a finished game by index, or None for current live game."""
        self.viewing_finished_idx = finished_idx
        if finished_idx is None:
            self.follow_live = True
            self.viewing_ply = None
        else:
            self.follow_live = False
            game = self.get_viewing_game()
            self.viewing_ply = len(game.moves) if game else 0

    def set_follow_live(self, enabled: bool) -> None:
        self.follow_live = enabled
        if enabled:
            self.viewing_finished_idx = None
            self.viewing_ply = None


# ----------------------------------------------------------------------
# Visualizer Rendering Component
# ----------------------------------------------------------------------

class TrainingVisualizer:
    """Renders the comprehensive reinforcement learning visualiser view."""

    def __init__(self, state: TrainingVizState, ui_scale: float = 1.0):
        self.state = state
        self.ui_scale = max(1.0, ui_scale)

        def s(v: float) -> int:
            return int(round(v * self.ui_scale))

        self.s = s

        # Dedicated board renderer for the smaller training view (360x360 board)
        self.board_size = s(360)
        self.board_x = s(20)
        self.board_y = s(210)
        self.board_renderer = ChessRenderer(
            board_size=360,
            board_offset_x=20,
            board_offset_y=210,
            ui_scale=self.ui_scale,
        )

        # Move list and game list scroll offsets
        self.move_list_scroll: int = 0
        self.game_list_scroll: int = 0

        # Clickable rect hit-tests recorded during draw
        self._move_row_rects: List[Tuple[pygame.Rect, int]] = []
        self._game_row_rects: List[Tuple[pygame.Rect, Optional[int]]] = []

    def _clip_text_to_width(self, font: pygame.font.Font, text: str, max_width: int) -> str:
        if max_width <= 0:
            return ""
        if font.size(text)[0] <= max_width:
            return text
        ell = "…"
        for n in range(len(text) - 1, 0, -1):
            clipped = text[:n] + ell
            if font.size(clipped)[0] <= max_width:
                return clipped
        return ell

    def _move_list_label(self, san: str, root_value_white: float) -> str:
        sign = "+" if root_value_white >= 0 else ""
        return f"{san} {sign}{root_value_white:.2f}"

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle clicks or scrolls inside the visualiser layout."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = event.pos

            # Check move row click
            for rect, ply in self._move_row_rects:
                if rect.collidepoint(pos):
                    self.state.follow_live = False
                    self.state.viewing_ply = ply
                    return True

            # Check game row click
            for rect, g_idx in self._game_row_rects:
                if rect.collidepoint(pos):
                    self.state.select_game(g_idx)
                    return True

        elif event.type == pygame.MOUSEWHEEL:
            # Scroll move list or game list
            m_pos = getattr(event, "pos", pygame.mouse.get_pos())
            s = self.s
            move_card = pygame.Rect(s(395), s(172), s(240), s(212))
            game_card = pygame.Rect(s(395), s(392), s(240), s(216))

            if move_card.collidepoint(m_pos):
                self.move_list_scroll = max(0, self.move_list_scroll - event.y * 2)
                return True
            elif game_card.collidepoint(m_pos):
                self.game_list_scroll = max(0, self.game_list_scroll - event.y * 2)
                return True

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                self.state.step_ply(-1)
                return True
            elif event.key == pygame.K_RIGHT:
                self.state.step_ply(1)
                return True
            elif event.key == pygame.K_HOME:
                self.state.go_to_start()
                return True
            elif event.key == pygame.K_END:
                self.state.go_to_end()
                return True

        return False

    def render(self, surface: pygame.Surface) -> None:
        """Main render loop for the training visualiser view."""
        self._move_row_rects.clear()
        self._game_row_rects.clear()

        # 1. Top Stage Strip (y=136..164)
        self._render_stage_strip(surface)

        # 2. Left Column: Board + Player badge (x=20..385)
        self._render_board_column(surface)

        # 3. Middle Column: Moves + Finished games (x=395..635)
        self._render_middle_column(surface)

        # 4. Right Column: Phase-Aware RL Explanation Panel (x=645..1020)
        self._render_right_column(surface)

    # ------------------------------------------------------------------
    # 1. Stage Strip (Self-Play -> Training -> Arena -> Promotion)
    # ------------------------------------------------------------------

    def _render_stage_strip(self, surface: pygame.Surface) -> None:
        s = self.s
        strip_rect = pygame.Rect(s(20), s(136), s(1000), s(28))
        pygame.draw.rect(surface, (20, 22, 26), strip_rect, border_radius=s(6))
        pygame.draw.rect(surface, COLOR_BORDER, strip_rect, width=1, border_radius=s(6))

        stages = [
            ("selfplay", "1  Self-play"),
            ("training", "2  Train network"),
            ("arena", "3  Arena test"),
            ("promoted", "4  Promote"),
        ]

        active_phase = self.state.phase
        # Map phase to active index
        stage_idx = 0
        if active_phase == "training":
            stage_idx = 1
        elif active_phase == "arena":
            stage_idx = 2
        elif active_phase in ("completed", "stopped") or self.state.last_gen_summary:
            stage_idx = 3

        col_w = strip_rect.width // 4
        for i, (k, label) in enumerate(stages):
            rx = strip_rect.x + i * col_w
            cell_rect = pygame.Rect(rx, strip_rect.y, col_w, strip_rect.height)
            is_active = (i == stage_idx)
            is_done = (i < stage_idx)

            if is_active:
                pygame.draw.rect(surface, (36, 60, 42), cell_rect, border_radius=s(4))
                pygame.draw.rect(surface, COLOR_PRIMARY, cell_rect, width=1, border_radius=s(4))
                text_color = COLOR_PRIMARY
            elif is_done:
                text_color = COLOR_TEXT
            else:
                text_color = COLOR_TEXT_MUTED

            txt = self.board_renderer.font_sm.render(label, True, text_color)
            surface.blit(txt, txt.get_rect(center=cell_rect.center))

    # ------------------------------------------------------------------
    # 2. Left Column: Board & telemetry header
    # ------------------------------------------------------------------

    def _render_board_column(self, surface: pygame.Surface) -> None:
        s = self.s
        board, move_rec = self.state.get_viewing_board_and_move()
        game = self.state.get_viewing_game()

        # Header card above board (y=172..204)
        hdr_rect = pygame.Rect(s(20), s(172), s(360), s(32))
        pygame.draw.rect(surface, COLOR_PANEL_CARD, hdr_rect, border_radius=s(6))
        pygame.draw.rect(surface, COLOR_BORDER, hdr_rect, width=1, border_radius=s(6))

        # Title: Game X of N (or finished game label)
        if game:
            prefix = "Arena" if game.phase == "arena" else "Game"
            g_lbl = f"{prefix} {game.game_idx}/{game.total_games}"
            if game.phase == "arena" and game.candidate_color:
                side = game.candidate_color.title()
                g_lbl += f" · Candidate: {side}"
            if self.state.viewing_finished_idx is not None:
                g_lbl = f"[Review] {g_lbl}"
        else:
            g_lbl = "Awaiting Game..."

        txt_g = self.board_renderer.font_sm.render(g_lbl, True, COLOR_TEXT)
        surface.blit(txt_g, (hdr_rect.x + s(10), hdr_rect.y + s(7)))

        balance = material_balance_white(board)
        mat_lbl = format_material_balance(board)
        if balance > 0:
            mat_color = (235, 235, 245)
        elif balance < 0:
            mat_color = (160, 160, 175)
        else:
            mat_color = COLOR_TEXT_MUTED
        txt_mat = self.board_renderer.font_sm.render(mat_lbl, True, mat_color)
        surface.blit(txt_mat, (hdr_rect.right - txt_mat.get_width() - s(10), hdr_rect.y + s(7)))

        # Draw Chessboard
        last_move = chess.Move(move_rec.from_sq, move_rec.to_sq) if move_rec else None

        # Build analysis object if we have candidates on current move
        analysis = None
        if move_rec and move_rec.candidates:
            cands = []
            for c in move_rec.candidates:
                c_mv = chess.Move.from_uci(c.uci) if c.uci else chess.Move.null()
                cands.append(
                    CandidateMoveAnalysis(
                        action=0,
                        move=c_mv,
                        san=c.san,
                        visits=c.visits,
                        visit_share=c.visit_share,
                        prior=c.prior,
                        q_value=c.q_value,
                    )
                )
            pv_moves = [chess.Move.from_uci(m) for m in move_rec.pv if m]
            analysis = MCTSAnalysis(
                root_value=move_rec.root_value,
                total_simulations=move_rec.sims,
                candidates=cands,
                principal_variation=pv_moves,
                pv_san=move_rec.pv_san,
            )

        self.board_renderer.draw_board(
            surface=surface,
            board=board,
            last_move=last_move,
            analysis=analysis,
        )

    # ------------------------------------------------------------------
    # 3. Middle Column: Move History + Finished Games List
    # ------------------------------------------------------------------

    def _render_middle_column(self, surface: pygame.Surface) -> None:
        s = self.s
        game = self.state.get_viewing_game()
        active_ply = self.state.get_viewing_ply()

        # Top Card: Move List (y=172..384, height 212)
        card_moves = pygame.Rect(s(395), s(172), s(240), s(212))
        pygame.draw.rect(surface, COLOR_PANEL_CARD, card_moves, border_radius=s(6))
        pygame.draw.rect(surface, COLOR_BORDER, card_moves, width=1, border_radius=s(6))

        # Header
        hdr_txt = self.board_renderer.font_md.render("Move List (Eval)", True, COLOR_TEXT)
        surface.blit(hdr_txt, (card_moves.x + s(10), card_moves.y + s(8)))
        pygame.draw.line(surface, COLOR_BORDER, (card_moves.x, card_moves.y + s(30)), (card_moves.right, card_moves.y + s(30)))

        # Move list items
        moves_clip = pygame.Rect(card_moves.x + s(6), card_moves.y + s(34), card_moves.width - s(12), card_moves.height - s(40))
        orig_clip = surface.get_clip()
        surface.set_clip(moves_clip)

        row_h = s(20)
        moves = game.moves if game else []
        total_rows = (len(moves) + 1) // 2

        # Auto-scroll if following live
        if self.state.follow_live and total_rows > 8:
            self.move_list_scroll = max(0, total_rows - 8)

        max_scroll = max(0, total_rows - 8)
        self.move_list_scroll = max(0, min(max_scroll, self.move_list_scroll))

        font = self.board_renderer.font_mono
        num_col_w = s(26)
        half_gap = s(4)
        half_w = max(s(48), (moves_clip.width - num_col_w - half_gap) // 2)
        white_x = moves_clip.x + num_col_w
        black_x = white_x + half_w + half_gap
        text_pad = s(2)
        label_max_w = half_w - text_pad * 2

        y = moves_clip.y - self.move_list_scroll * row_h
        for row_i in range(total_rows):
            w_idx = row_i * 2
            b_idx = w_idx + 1

            move_num = row_i + 1
            num_surf = font.render(f"{move_num:>3}.", True, COLOR_TEXT_MUTED)
            surface.blit(num_surf, (moves_clip.x + s(2), y + s(2)))

            def draw_half_move(
                move_rec: MoveRecord,
                col_x: int,
                half_width: int,
            ) -> None:
                is_active = active_ply == move_rec.ply
                col = COLOR_PRIMARY if is_active else COLOR_TEXT
                cell = pygame.Rect(col_x, y, half_width, row_h)
                if is_active:
                    pygame.draw.rect(surface, (36, 60, 42), cell, border_radius=s(3))
                self._move_row_rects.append((cell, move_rec.ply))
                label = self._clip_text_to_width(
                    font,
                    self._move_list_label(move_rec.san, move_rec.root_value_white),
                    label_max_w,
                )
                surf = font.render(label, True, col)
                surface.blit(surf, (cell.x + text_pad, y + s(2)))

            if w_idx < len(moves):
                draw_half_move(moves[w_idx], white_x, half_w)
            if b_idx < len(moves):
                draw_half_move(moves[b_idx], black_x, half_w)

            y += row_h

        surface.set_clip(orig_clip)

        # Bottom Card: Finished Games in Gen (y=392..608, height 216)
        card_games = pygame.Rect(s(395), s(392), s(240), s(216))
        pygame.draw.rect(surface, COLOR_PANEL_CARD, card_games, border_radius=s(6))
        pygame.draw.rect(surface, COLOR_BORDER, card_games, width=1, border_radius=s(6))

        n_finished = len(self.state.finished_games)
        n_sp = sum(1 for g in self.state.finished_games if g.phase == "selfplay")
        n_ar = sum(1 for g in self.state.finished_games if g.phase == "arena")
        has_current = self.state.current_game is not None
        g_title = f"Games ({n_sp} self-play, {n_ar} arena)"
        hdr_g = self.board_renderer.font_md.render(g_title, True, COLOR_TEXT)
        surface.blit(hdr_g, (card_games.x + s(10), card_games.y + s(8)))
        pygame.draw.line(surface, COLOR_BORDER, (card_games.x, card_games.y + s(30)), (card_games.right, card_games.y + s(30)))

        games_clip = pygame.Rect(card_games.x + s(6), card_games.y + s(34), card_games.width - s(12), card_games.height - s(40))
        surface.set_clip(games_clip)

        g_row_h = s(24)
        total_g = n_finished + (1 if has_current else 0)
        max_g_scroll = max(0, total_g - 7)
        self.game_list_scroll = max(0, min(max_g_scroll, self.game_list_scroll))

        gy = games_clip.y - self.game_list_scroll * g_row_h

        if has_current:
            cg = self.state.current_game
            is_viewing = self.state.viewing_finished_idx is None
            row_rect = pygame.Rect(games_clip.x, gy, games_clip.width, g_row_h - s(2))
            if is_viewing:
                pygame.draw.rect(surface, (40, 65, 48), row_rect, border_radius=s(4))
                pygame.draw.rect(surface, COLOR_PRIMARY, row_rect, width=1, border_radius=s(4))
            else:
                pygame.draw.rect(surface, (32, 42, 36), row_rect, border_radius=s(4))
                pygame.draw.rect(surface, COLOR_PRIMARY_DARK, row_rect, width=1, border_radius=s(4))

            self._game_row_rects.append((row_rect, None))

            if cg.is_finished:
                status = "last played"
            else:
                status = f"{len(cg.moves)} plies"
            if cg.phase == "arena":
                cand_side = f"  Cand:{cg.candidate_color[0].upper()}" if cg.candidate_color else ""
                row_txt = f"▶ Arena {cg.game_idx}/{cg.total_games}{cand_side}  {status}"
            else:
                row_txt = f"▶ Self-play {cg.game_idx}/{cg.total_games}  {status}"
            col_t = COLOR_PRIMARY if is_viewing else COLOR_TEXT
            surf_row = self.board_renderer.font_sm.render(row_txt, True, col_t)
            surface.blit(surf_row, (row_rect.x + s(6), gy + s(3)))
            gy += g_row_h

        for idx, g_rec in enumerate(self.state.finished_games):
            is_viewing = (self.state.viewing_finished_idx == idx)
            row_rect = pygame.Rect(games_clip.x, gy, games_clip.width, g_row_h - s(2))
            if is_viewing:
                pygame.draw.rect(surface, (40, 65, 48), row_rect, border_radius=s(4))
                pygame.draw.rect(surface, COLOR_PRIMARY, row_rect, width=1, border_radius=s(4))
            else:
                pygame.draw.rect(surface, (32, 34, 38), row_rect, border_radius=s(4))

            self._game_row_rects.append((row_rect, idx))

            pts_lbl = "1/2" if g_rec.result == "1/2-1/2" else g_rec.result
            if g_rec.phase == "arena":
                cand_side = f" C:{g_rec.candidate_color[0].upper()}" if g_rec.candidate_color else ""
                row_txt = f"A{g_rec.game_idx}/{g_rec.total_games}  {pts_lbl:<6}{cand_side} {g_rec.plies}p"
            else:
                row_txt = f"#{g_rec.game_idx}  {pts_lbl:<6} {g_rec.plies} plies"
            col_t = COLOR_PRIMARY if is_viewing else COLOR_TEXT
            surf_row = self.board_renderer.font_sm.render(row_txt, True, col_t)
            surface.blit(surf_row, (row_rect.x + s(8), gy + s(3)))

            gy += g_row_h

        surface.set_clip(orig_clip)

    # ------------------------------------------------------------------
    # 4. Right Column: Phase-Aware RL Telemetry Panel
    # ------------------------------------------------------------------

    def _render_right_column(self, surface: pygame.Surface) -> None:
        s = self.s
        card_right = pygame.Rect(s(645), s(172), s(375), s(436))
        pygame.draw.rect(surface, COLOR_PANEL_CARD, card_right, border_radius=s(6))
        pygame.draw.rect(surface, COLOR_BORDER, card_right, width=1, border_radius=s(6))

        if self.state.phase == "training":
            self._render_training_panel(surface, card_right)
        elif self.state.phase == "arena":
            self._render_arena_panel(surface, card_right)
        else:
            self._render_selfplay_panel(surface, card_right)

    def _render_selfplay_panel(self, surface: pygame.Surface, card: pygame.Rect) -> None:
        s = self.s
        _, move_rec = self.state.get_viewing_board_and_move()
        game = self.state.get_viewing_game()

        x = card.x + s(14)
        y = card.y + s(12)

        # Title
        title_s = self.board_renderer.font_lg.render("MCTS Policy & Value Signal", True, COLOR_TEXT)
        surface.blit(title_s, (x, y))
        y += s(26)

        # Win Expectancy Bar
        root_val_white = move_rec.root_value_white if move_rec else 0.0
        white_pct = (root_val_white + 1.0) / 2.0
        val_txt = f"Win Expectancy: {white_pct * 100:.1f}% White (Q={root_val_white:+.2f})"
        stat_s = self.board_renderer.font_sm.render(val_txt, True, COLOR_TEXT)
        surface.blit(stat_s, (x, y))
        y += s(18)

        bar_w = card.width - s(28)
        bar_h = s(14)
        pygame.draw.rect(surface, (40, 40, 40), (x, y, bar_w, bar_h), border_radius=s(3))
        w_px = int(bar_w * white_pct)
        if w_px > 0:
            pygame.draw.rect(surface, (235, 235, 235), (x, y, w_px, bar_h), border_radius=s(3))
        y += s(24)

        # MCTS Candidates: Network Prior vs MCTS Search Visits
        sub_title = self.board_renderer.font_md.render("Network Prior P(a) vs MCTS Visits N(a)", True, COLOR_TEXT)
        surface.blit(sub_title, (x, y))
        y += s(20)

        # Legend
        swatch = s(10)
        for i, (label, color) in enumerate((("NN Prior P", (130, 190, 240)), ("MCTS Visits π", COLOR_PRIMARY))):
            lx = x + i * s(110)
            pygame.draw.rect(surface, color, (lx, y + s(4), swatch, swatch))
            surface.blit(self.board_renderer.font_sm.render(label, True, color), (lx + swatch + s(5), y))
        y += s(20)

        cands = move_rec.candidates if move_rec else []
        if cands:
            for idx, c in enumerate(cands[:4]):
                row_txt = f"#{idx + 1} {c.san:<6} Q={c.q_value:+.2f}"
                txt_row = self.board_renderer.font_mono.render(row_txt, True, COLOR_TEXT)
                surface.blit(txt_row, (x, y))
                y += s(16)

                # Dual bars: top is Prior P(a) (cyan), bottom is Visit Share (green)
                bar_max_w = bar_w - s(10)

                # Prior bar
                p_w = int(bar_max_w * min(1.0, c.prior))
                pygame.draw.rect(surface, (45, 50, 60), (x, y, bar_max_w, s(6)), border_radius=s(2))
                if p_w > 0:
                    pygame.draw.rect(surface, (100, 180, 240), (x, y, p_w, s(6)), border_radius=s(2))
                y += s(8)

                # Visits bar
                v_w = int(bar_max_w * min(1.0, c.visit_share))
                pygame.draw.rect(surface, (45, 50, 60), (x, y, bar_max_w, s(6)), border_radius=s(2))
                if v_w > 0:
                    pygame.draw.rect(surface, COLOR_PRIMARY, (x, y, v_w, s(6)), border_radius=s(2))
                y += s(12)
        else:
            none_c = self.board_renderer.font_sm.render("Awaiting move search...", True, COLOR_TEXT_MUTED)
            surface.blit(none_c, (x, y))
            y += s(60)

        # Eval Sparkline by Ply
        spark_title = self.board_renderer.font_md.render("Game Evaluation Curve (White Eval)", True, COLOR_TEXT)
        surface.blit(spark_title, (x, y))
        y += s(18)

        spark_rect = pygame.Rect(x, y, bar_w, s(50))
        pygame.draw.rect(surface, (25, 27, 31), spark_rect, border_radius=s(4))
        pygame.draw.line(surface, COLOR_BORDER, (spark_rect.x, spark_rect.centery), (spark_rect.right, spark_rect.centery), 1)

        if game and len(game.moves) > 1:
            pts = []
            for i, m in enumerate(game.moves):
                px = spark_rect.x + int((spark_rect.width - 2) * (i / (len(game.moves) - 1)))
                py = spark_rect.centery - int((spark_rect.height / 2 - 4) * max(-1.0, min(1.0, m.root_value_white)))
                pts.append((px, py))
            if len(pts) >= 2:
                pygame.draw.lines(surface, COLOR_PRIMARY, False, pts, width=2)
        y += s(58)

        # Game Result / Replay target banner
        if game and game.is_finished:
            banner_rect = pygame.Rect(x, y, bar_w, s(52))
            pygame.draw.rect(surface, (28, 48, 34), banner_rect, border_radius=s(4))
            pygame.draw.rect(surface, COLOR_PRIMARY, banner_rect, width=1, border_radius=s(4))

            z_lbl = "+1.0" if game.white_val > 0 else ("-1.0" if game.white_val < 0 else "0.0")
            b_txt1 = f"Outcome: {game.result} • Target z={z_lbl} stamped on {game.samples_count} pos"
            b_txt2 = f"Replay Buffer: {self.state.replay_buffer_size:,} states • {game.termination_reason}"

            s1 = self.board_renderer.font_sm.render(b_txt1, True, COLOR_PRIMARY)
            s2 = self.board_renderer.font_sm.render(b_txt2, True, COLOR_TEXT)
            surface.blit(s1, (banner_rect.x + s(8), banner_rect.y + s(6)))
            surface.blit(s2, (banner_rect.x + s(8), banner_rect.y + s(26)))

    def _render_training_panel(self, surface: pygame.Surface, card: pygame.Rect) -> None:
        s = self.s
        x = card.x + s(14)
        y = card.y + s(12)

        title_s = self.board_renderer.font_lg.render("Neural Network Optimization", True, COLOR_TEXT)
        surface.blit(title_s, (x, y))
        y += s(26)

        history = self.state.loss_history
        cur_step = history[-1] if history else None

        if cur_step:
            step_lbl = f"Step {cur_step.step}/{cur_step.total_steps} • Total Loss: {cur_step.loss_total:.4f}"
            s_txt = self.board_renderer.font_sm.render(step_lbl, True, COLOR_PRIMARY)
            surface.blit(s_txt, (x, y))
            y += s(18)

            detail_lbl = f"Policy Loss: {cur_step.loss_policy:.4f} • Value Loss: {cur_step.loss_value:.4f} • LR: {cur_step.lr:.5f}"
            d_txt = self.board_renderer.font_sm.render(detail_lbl, True, COLOR_TEXT)
            surface.blit(d_txt, (x, y))
            y += s(24)
        else:
            waiting = self.board_renderer.font_sm.render("Awaiting training steps...", True, COLOR_TEXT_MUTED)
            surface.blit(waiting, (x, y))
            y += s(42)

        bar_w = card.width - s(28)

        # Plot 1: Policy Loss Curve
        p_title = self.board_renderer.font_md.render("Policy Loss - Cross Entropy vs MCTS π", True, COLOR_TEXT)
        surface.blit(p_title, (x, y))
        y += s(18)

        plot1_rect = pygame.Rect(x, y, bar_w, s(85))
        self._render_curve(surface, plot1_rect, [h.loss_policy for h in history], color=(100, 180, 240), label="Policy Loss")
        y += s(95)

        # Plot 2: Value Loss Curve
        v_title = self.board_renderer.font_md.render("Value Loss - MSE vs Game Outcome z", True, COLOR_TEXT)
        surface.blit(v_title, (x, y))
        y += s(18)

        plot2_rect = pygame.Rect(x, y, bar_w, s(85))
        self._render_curve(surface, plot2_rect, [h.loss_value for h in history], color=COLOR_PRIMARY, label="Value Loss")
        y += s(95)

        # Explanation Footer
        loss_formula = "Objective: L = (z - v)^2 - π^T log(p) + c||θ||^2"
        f_surf = self.board_renderer.font_mono.render(loss_formula, True, COLOR_TEXT_MUTED)
        surface.blit(f_surf, (x, y + s(4)))

    def _render_arena_panel(self, surface: pygame.Surface, card: pygame.Rect) -> None:
        s = self.s
        x = card.x + s(14)
        y = card.y + s(12)

        title_s = self.board_renderer.font_lg.render("Arena Gating: Candidate vs Champion", True, COLOR_TEXT)
        surface.blit(title_s, (x, y))
        y += s(22)
        alt_s = self.board_renderer.font_sm.render(
            "Colors alternate: candidate is White on games 1,3,5… and Black on 2,4,6…",
            True,
            COLOR_TEXT_MUTED,
        )
        surface.blit(alt_s, (x, y))
        y += s(22)

        live = self.state.current_game
        if live and live.phase == "arena" and live.candidate_color and not live.is_finished:
            live_txt = (
                f"Live game {live.game_idx}/{live.total_games}: "
                f"{live.white} vs {live.black} — candidate plays {live.candidate_color.title()}"
            )
            live_s = self.board_renderer.font_sm.render(live_txt, True, COLOR_PRIMARY)
            surface.blit(live_s, (x, y))
            y += s(20)

        # Candidate scoreboard card
        bar_w = card.width - s(28)
        score_rect = pygame.Rect(x, y, bar_w, s(85))
        pygame.draw.rect(surface, (25, 27, 31), score_rect, border_radius=s(6))
        pygame.draw.rect(surface, COLOR_BORDER, score_rect, width=1, border_radius=s(6))

        w, d, l = self.state.arena_candidate_wins, self.state.arena_candidate_draws, self.state.arena_candidate_losses
        tot = w + d + l
        score_pct = (w + 0.5 * d) / max(1, tot) * 100.0

        lbl_score = self.board_renderer.font_title.render(f"{w}W - {d}D - {l}L  ({score_pct:.1f}%)", True, COLOR_PRIMARY)
        surface.blit(lbl_score, (score_rect.x + s(16), score_rect.y + s(14)))

        arena_total = max(1, self.state.arena_total_games)
        t_lbl = (
            f"Required Threshold: {self.state.arena_threshold * 100:.0f}% win rate "
            f"({tot}/{arena_total} arena games played)"
        )
        sub_s = self.board_renderer.font_sm.render(t_lbl, True, COLOR_TEXT_MUTED)
        surface.blit(sub_s, (score_rect.x + s(16), score_rect.y + s(48)))
        y += s(98)

        # Threshold progress bar comparison
        thresh_title = self.board_renderer.font_md.render("Candidate Win Rate vs Promotion Gate", True, COLOR_TEXT)
        surface.blit(thresh_title, (x, y))
        y += s(20)

        t_bar_h = s(18)
        pygame.draw.rect(surface, (40, 42, 48), (x, y, bar_w, t_bar_h), border_radius=s(4))

        # Fill actual score
        score_w = int(bar_w * (score_pct / 100.0))
        if score_w > 0:
            fill_c = COLOR_PRIMARY if score_pct >= (self.state.arena_threshold * 100) else (230, 160, 60)
            pygame.draw.rect(surface, fill_c, (x, y, score_w, t_bar_h), border_radius=s(4))

        # Draw 55% marker line
        marker_x = x + int(bar_w * self.state.arena_threshold)
        pygame.draw.line(surface, (255, 255, 255), (marker_x, y - s(2)), (marker_x, y + t_bar_h + s(2)), width=2)
        y += s(30)

        # Promotion status card
        if self.state.last_gen_summary:
            promoted = self.state.last_gen_summary.get("promoted", False)
            res_rect = pygame.Rect(x, y, bar_w, s(60))
            bg_c = (28, 52, 36) if promoted else (54, 32, 34)
            border_c = COLOR_PRIMARY if promoted else (220, 80, 80)
            pygame.draw.rect(surface, bg_c, res_rect, border_radius=s(6))
            pygame.draw.rect(surface, border_c, res_rect, width=1, border_radius=s(6))

            status_txt = "PROMOTED TO CHAMPION!" if promoted else "CANDIDATE REJECTED"
            stat_col = COLOR_PRIMARY if promoted else (240, 110, 110)
            t_surf = self.board_renderer.font_lg.render(status_txt, True, stat_col)
            surface.blit(t_surf, (res_rect.x + s(14), res_rect.y + s(10)))

            champ_txt = f"Champion is now: {self.state.last_gen_summary.get('champion_name')}"
            c_surf = self.board_renderer.font_sm.render(champ_txt, True, COLOR_TEXT)
            surface.blit(c_surf, (res_rect.x + s(14), res_rect.y + s(34)))

    def _render_curve(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        values: List[float],
        color: Tuple[int, int, int],
        label: str,
    ) -> None:
        """Helper to render a clean mini line chart with min/max scale labels."""
        s = self.s
        pygame.draw.rect(surface, (25, 27, 31), rect, border_radius=s(4))
        pygame.draw.rect(surface, COLOR_BORDER, rect, width=1, border_radius=s(4))

        if not values:
            none_s = self.board_renderer.font_sm.render("No loss values yet", True, COLOR_TEXT_MUTED)
            surface.blit(none_s, (rect.x + s(10), rect.centery - s(7)))
            return

        min_v = min(values)
        max_v = max(values)
        span = max(1e-6, max_v - min_v)

        pts = []
        for i, val in enumerate(values):
            px = rect.x + int((rect.width - 4) * (i / max(1, len(values) - 1)))
            py = rect.bottom - s(4) - int((rect.height - s(8)) * ((val - min_v) / span))
            pts.append((px, py))

        if len(pts) >= 2:
            pygame.draw.lines(surface, color, False, pts, width=2)
        elif pts:
            pygame.draw.circle(surface, color, pts[0], 3)

        # Min and Max text labels
        mx_s = self.board_renderer.font_sm.render(f"{max_v:.3f}", True, COLOR_TEXT_MUTED)
        mn_s = self.board_renderer.font_sm.render(f"{min_v:.3f}", True, COLOR_TEXT_MUTED)
        surface.blit(mx_s, (rect.right - mx_s.get_width() - s(6), rect.y + s(3)))
        surface.blit(mn_s, (rect.right - mn_s.get_width() - s(6), rect.bottom - mn_s.get_height() - s(2)))
