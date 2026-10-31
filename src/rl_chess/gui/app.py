"""Unified Pygame dashboard: Play Computer, Training, Benchmark Elo, Statistics."""

from __future__ import annotations

import datetime
import os
import queue
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chess
import chess.pgn
import pygame
import torch

from rl_chess.benchmark import run_benchmark
from rl_chess.config import DEFAULT_CONFIG_PATH, get_paths, load_config
from rl_chess.gui.display import HiDPIWindow
from rl_chess.gui.play import PlaySession
from rl_chess.gui.renderer import ChessRenderer
from rl_chess.gui.train_viz import TrainingVisualizer, TrainingVizState
from rl_chess.gui.widgets import COLOR_BG, Button, CycleButton, JobPanel, TabBar
from rl_chess.gui.workers import BackgroundWorker
from rl_chess.ledger import Ledger
from rl_chess.game import result_string
from rl_chess.mcts import MCTS, MCTSAnalysis
from rl_chess.model import load_weights, model_from_config
from rl_chess.train.loop import run_training
from rl_chess.train.reset import reset_agent_to_gen_zero

CHAMPION_OPTION = "Champion (Latest)"
LOGICAL_WINDOW_SIZE = (1040, 620)


class ChessApp:
    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG_PATH,
        checkpoint_target: str = "",
        human_color_str: str = "white",
    ):
        pygame.init()
        self._hidpi: Optional[HiDPIWindow] = None
        if os.environ.get("SDL_VIDEODRIVER") != "dummy":
            try:
                self._hidpi = HiDPIWindow("Chess", LOGICAL_WINDOW_SIZE, background=COLOR_BG)
            except (OSError, RuntimeError, AttributeError):
                self._hidpi = None
        if self._hidpi is not None and self._hidpi.scale > 1.0:
            self.renderer = ChessRenderer(ui_scale=self._hidpi.scale)
            self.screen = self._hidpi.surface
        else:
            if self._hidpi is not None:
                self._hidpi.close()
                self._hidpi = None
            self.renderer = ChessRenderer(ui_scale=1.0)
            pygame.display.set_caption("Chess")
            headless = os.environ.get("SDL_VIDEODRIVER") == "dummy"
            flags = 0 if headless else pygame.SCALED | pygame.RESIZABLE
            self.screen = pygame.display.set_mode(LOGICAL_WINDOW_SIZE, flags)
        r = self.renderer
        self.clock = pygame.time.Clock()

        self.config_path = config_path
        self.cfg = load_config(config_path)
        paths = get_paths(self.cfg)
        self.ledger = Ledger(paths["ledger_db"])
        self.pgns_dir = Path(paths["pgns_dir"])
        self.pgns_dir.mkdir(parents=True, exist_ok=True)
        self.simulations = self.cfg.get("mcts", {}).get("num_simulations", 64)

        self.tab_bar = TabBar(
            r.rect(0, 0, 1040, 44),
            tabs=[("play", "Play Computer"), ("train", "Training"), ("benchmark", "Benchmark Elo"), ("stats", "Statistics")],
            initial_tab="play",
            on_tab_change=self._on_tab_change,
        )
        self.current_tab = "play"
        self.worker_queue: queue.Queue = queue.Queue()

        # Play tab
        self.net = model_from_config(self.cfg)
        self.checkpoint_id: Optional[int] = None
        self.agent_name = "Random-Agent"
        self._load_agent_checkpoint(checkpoint_target)
        self.mcts = MCTS(self.net, device=self.cfg.get("device", "auto"))
        human_color = chess.WHITE if human_color_str.lower() == "white" else chess.BLACK
        self.session = PlaySession(human_color, drag_threshold=max(6, r.s(10)))
        self.latest_analysis: Optional[MCTSAnalysis] = None
        self.is_agent_thinking = False
        self.agent_move_queue: queue.Queue = queue.Queue()
        self.game_logged = False
        self.agent_weights_loading = False
        self.agent_weights_message = "Weights ready"
        self.agent_weights_flash_until_ms = 0
        self.agent_reload_queue: queue.Queue = queue.Queue()
        self._agent_reload_thread: Optional[threading.Thread] = None
        self._agent_name_before_job: str = self.agent_name

        # Training and benchmark tabs
        self.latest_bench_elo: Optional[Dict[str, Any]] = None
        self.train_panel = self._make_train_panel()
        self.bench_panel = self._make_bench_panel()
        self.panels = {"train": self.train_panel, "bench": self.bench_panel}

        # Training Visualizer & state
        self.train_viz_state = TrainingVizState()
        self.train_visualizer = TrainingVisualizer(self.train_viz_state, ui_scale=self.renderer.ui_scale)
        self.train_view_mode: str = "visualiser"

        self.train_view_btn = Button(r.rect(410, 92, 125, 36), "View: Board", on_click=self._toggle_train_view)
        self.train_pace_btn = Button(r.rect(545, 92, 100, 36), "Speed: Max", on_click=self._toggle_train_pace)
        self.train_live_btn = Button(r.rect(655, 92, 90, 36), "Live", on_click=self._on_train_live_click)
        self.train_prev_btn = Button(r.rect(755, 92, 40, 36), "<", on_click=lambda: self.train_viz_state.step_ply(-1))
        self.train_next_btn = Button(r.rect(805, 92, 40, 36), ">", on_click=lambda: self.train_viz_state.step_ply(1))
        self.train_end_btn = Button(r.rect(855, 92, 45, 36), ">|", on_click=self._on_train_end_click)

        # Statistics tab
        self.stats_summary: Dict[str, Any] = {}
        self.stats_elo_history: List[Dict[str, Any]] = []
        self.stats_checkpoints: List[Any] = []
        self.stats_refresh_btn = Button(r.rect(880, 54, 120, 32), "Refresh Stats", on_click=self._refresh_stats)
        self._refresh_stats()

    def _event_in_render_pixels(self, event: pygame.event.Event) -> pygame.event.Event:
        """Map window-point mouse coordinates to the high-resolution, letterboxed Surface."""
        if self._hidpi is None:
            return event
        attrs = event.dict.copy()
        if event.type == pygame.MOUSEWHEEL:
            attrs["pos"] = pygame.mouse.get_pos()
        elif not hasattr(event, "pos"):
            return event
        attrs["pos"] = self._hidpi.to_surface(attrs["pos"])
        if "rel" in attrs:
            k = self._hidpi.surface_per_point()
            attrs["rel"] = tuple(int(round(value * k)) for value in attrs["rel"])
        return pygame.event.Event(event.type, attrs)

    def _present(self) -> None:
        if self._hidpi is not None:
            self._hidpi.present()
        else:
            pygame.display.flip()

    # -- job panels ------------------------------------------------------

    def _job_panel(self, kind: str, run_fn, selectors: List[CycleButton], action_x: int, **kwargs: Any) -> JobPanel:
        r = self.renderer
        return JobPanel(
            worker=BackgroundWorker(kind, run_fn, self.worker_queue),
            selectors=selectors,
            action_rect=r.rect(action_x, 92, 160, 36),
            progress_rect=r.rect(20, 140, 1000, 24),
            log_rect=r.rect(20, 176, 1000, 424),
            line_height=r.s(18),
            **kwargs,
        )

    def _make_train_panel(self) -> JobPanel:
        r = self.renderer
        gens = CycleButton(r.rect(20, 92, 90, 36), "Gens", ["1", "2", "5", "10"])
        self.train_games_btn = CycleButton(
            r.rect(120, 92, 110, 36),
            "Games",
            ["5", "10", "25", "50"],
            initial_index=2,
        )

        def make_job() -> Tuple[str, Dict[str, Any]]:
            n = int(gens.value)
            games_n = int(self.train_games_btn.value)
            return (
                f"Starting training ({n} gen(s), {games_n} self-play games before each weight update)...",
                {
                    "config_path": self.config_path,
                    "generations": n,
                    "games_per_generation": games_n,
                },
            )

        def describe(d: Dict[str, Any]) -> Tuple[float, str]:
            phase = d.get("phase", "")
            if phase == "Promoted":
                name = d.get("champion_name", "champion")
                return 1.0, f"Champion promoted → {name}"
            return d.get("fraction", 0.0), (
                f"Gen {d.get('generation', 0)}: {phase} ({d.get('current', 0)}/{d.get('total', 0)})"
            )

        self.train_reset_btn = Button(
            r.rect(900, 54, 110, 32),
            "Reset Agent",
            on_click=self._reset_training_agent,
            color=(120, 55, 55),
            hover_color=(150, 70, 70),
        )

        return self._job_panel(
            "train", run_training, [gens, self.train_games_btn], 240,
            start_text="Start", stop_text="Stop", idle_label="",
            make_job=make_job, describe_progress=describe,
            on_started=self._on_training_started,
            on_finished=self._on_training_finished,
        )

    def _make_bench_panel(self) -> JobPanel:
        r = self.renderer
        checkpoint = CycleButton(r.rect(20, 92, 220, 36), "Checkpoint", self._checkpoint_options())
        games = CycleButton(r.rect(255, 92, 170, 36), "Games / Rung", ["5 games", "10 games", "20 games"], initial_index=1)
        self.bench_checkpoint_btn = checkpoint

        def make_job() -> Tuple[str, Dict[str, Any]]:
            n = int(games.value.split()[0])
            target = "" if checkpoint.value == CHAMPION_OPTION else checkpoint.value
            return f"Starting Stockfish benchmark ({n} games/rung)...", {
                "config_path": self.config_path, "checkpoint_target": target, "games_per_opponent": n,
            }

        def describe(d: Dict[str, Any]) -> Tuple[float, str]:
            return d.get("fraction", 0.0), (
                f"Testing vs {d.get('rung', '')} ({d.get('completed', 0)}/{d.get('total', 0)}) - "
                f"{d.get('wins', 0)}W {d.get('draws', 0)}D {d.get('losses', 0)}L"
            )

        return self._job_panel(
            "bench", run_benchmark, [checkpoint, games], 440,
            start_text="Run Gauntlet", stop_text="Stop Gauntlet", idle_label="Ready to benchmark",
            make_job=make_job, describe_progress=describe, on_result=self._on_bench_result,
        )

    def _checkpoint_options(self) -> List[str]:
        return [CHAMPION_OPTION] + [c.name for c in self.ledger.list_checkpoints()]

    def _toggle_train_view(self) -> None:
        if self.train_view_mode == "visualiser":
            self.train_view_mode = "log"
            self.train_view_btn.text = "View: Logs"
        else:
            self.train_view_mode = "visualiser"
            self.train_view_btn.text = "View: Board"

    def _toggle_train_pace(self) -> None:
        mode = self.train_viz_state.cycle_pace()
        self.train_pace_btn.text = f"Speed: {mode}"
        delay = self.train_viz_state.pace_delay_sec
        if hasattr(self.train_panel.worker, "set_pace_delay"):
            self.train_panel.worker.set_pace_delay(delay)

    def _sync_train_live_btn(self) -> None:
        if self.train_viz_state.viewing_finished_idx is not None:
            self.train_live_btn.text = "Resume"
        elif self.train_viz_state.follow_live:
            self.train_live_btn.text = "Live"
        else:
            self.train_live_btn.text = "Paused"

    def _toggle_train_live(self) -> None:
        new_state = not self.train_viz_state.follow_live
        self.train_viz_state.set_follow_live(new_state)
        self._sync_train_live_btn()

    def _on_train_live_click(self) -> None:
        if self.train_viz_state.viewing_finished_idx is not None:
            self.train_viz_state.select_game(None)
        else:
            self._toggle_train_live()

    def _on_train_end_click(self) -> None:
        if self.train_viz_state.viewing_finished_idx is not None:
            self.train_viz_state.select_game(None)
        else:
            self.train_viz_state.go_to_end()
        self._sync_train_live_btn()

    def _on_training_started(self) -> None:
        self._agent_name_before_job = self.agent_name
        self._reset_train_visualizer_state()

    def _reset_train_visualizer_state(self) -> None:
        self.train_viz_state = TrainingVizState()
        self.train_visualizer.state = self.train_viz_state
        self._sync_train_live_btn()

    def _sync_play_agent_from_ledger(self) -> None:
        self._load_agent_checkpoint("")
        self.mcts = MCTS(self.net, device=self.cfg.get("device", "auto"))
        self.new_game()
        self.bench_checkpoint_btn.set_options(self._checkpoint_options())

    def _reset_training_agent(self) -> None:
        if self.train_panel.worker.is_running():
            self.train_panel.log.append("Stop training before resetting the agent.")
            return
        try:
            paths = get_paths(self.cfg)
            reset_agent_to_gen_zero(self.cfg, paths, self.ledger)
            self._reset_train_visualizer_state()
            self._sync_play_agent_from_ledger()
            self._refresh_stats()
            msg = "Agent reset: replay buffer, ledger, and all checkpoints cleared."
            self.agent_weights_message = msg
            self.agent_weights_flash_until_ms = pygame.time.get_ticks() + 8000
            self.train_panel.log.append(msg)
        except Exception as exc:
            self.train_panel.log.append(f"Reset failed: {exc}")

    def _on_training_finished(self, status: str) -> None:
        self._refresh_stats()
        self.train_panel.log.append("Syncing Play tab with ledger champion...")
        self.train_panel.progress.set_progress(
            self.train_panel.progress.progress,
            label="Loading champion weights into Play tab...",
            indeterminate=True,
        )
        self._reload_agent_weights_async(status)

    def _on_bench_result(self, result: Dict[str, Any]) -> None:
        self.latest_bench_elo = result
        self._refresh_stats()

    def _on_tab_change(self, tab_id: str) -> None:
        self.current_tab = tab_id
        if tab_id == "stats":
            self._refresh_stats()
        elif tab_id == "benchmark":
            self.bench_checkpoint_btn.set_options(self._checkpoint_options())

    def _refresh_stats(self) -> None:
        try:
            self.stats_summary = self.ledger.get_stats_summary()
            self.stats_elo_history = self.ledger.get_elo_history()
            self.stats_checkpoints = self.ledger.list_checkpoints()
            if self.stats_summary.get("latest_elo"):
                self.latest_bench_elo = self.stats_summary["latest_elo"]
        except Exception as e:
            print(f"Error refreshing stats: {e}")

    # -- play tab --------------------------------------------------------

    def _set_random_play_agent(self) -> None:
        self.net = model_from_config(self.cfg)
        self.net.eval()
        self.agent_name = "Random-Agent"
        self.checkpoint_id = None
        self.agent_weights_message = "Random initial weights (no checkpoint)"

    def _load_agent_checkpoint(self, target: str) -> None:
        rec = self.ledger.resolve_checkpoint(target)
        if rec and Path(rec.path).exists():
            load_weights(self.net, rec.path)
            self.agent_name = f"{rec.name} (Gen {rec.generation})"
            self.checkpoint_id = rec.id
            self.agent_weights_message = f"Loaded {self.agent_name}"
        else:
            self._set_random_play_agent()

    def _reload_agent_weights_async(self, run_status: str) -> None:
        if self._agent_reload_thread is not None and self._agent_reload_thread.is_alive():
            return
        self.agent_weights_loading = True
        self.agent_weights_message = "Loading champion checkpoint..."
        prev_name = self._agent_name_before_job

        def work() -> None:
            try:
                rec = self.ledger.resolve_checkpoint("")
                if rec and Path(rec.path).exists():
                    state = torch.load(rec.path, map_location="cpu")
                    self.agent_reload_queue.put(("ok", rec, state, prev_name, run_status))
                else:
                    self.agent_reload_queue.put(("missing", None, None, prev_name, run_status))
            except Exception as exc:
                self.agent_reload_queue.put(("error", str(exc), None, prev_name, run_status))

        self._agent_reload_thread = threading.Thread(target=work, daemon=True)
        self._agent_reload_thread.start()

    def _poll_agent_reload(self) -> None:
        try:
            item = self.agent_reload_queue.get_nowait()
        except queue.Empty:
            return

        kind = item[0]
        self.agent_weights_loading = False
        run_status = item[4] if len(item) > 4 else "completed"
        panel = self.train_panel

        if kind == "ok":
            rec, state, prev_name = item[1], item[2], item[3]
            self.net.load_state_dict(state)
            self.net.eval()
            self.agent_name = f"{rec.name} (Gen {rec.generation})"
            self.checkpoint_id = rec.id
            self.mcts.reset()
            changed = prev_name != self.agent_name
            if changed:
                summary = f"Weights updated: {prev_name} → {self.agent_name}"
            else:
                summary = f"Weights unchanged ({self.agent_name})"
            self.agent_weights_message = summary
            panel.log.append(summary)
            panel.progress.set_progress(1.0 if run_status == "completed" else panel.progress.progress, label=summary)
        elif kind == "missing":
            self.agent_weights_message = "No champion checkpoint found"
            panel.log.append(self.agent_weights_message)
            panel.progress.set_progress(panel.progress.progress, label=self.agent_weights_message)
        else:
            err = item[1]
            self.agent_weights_message = f"Failed to load weights: {err}"
            panel.log.append(self.agent_weights_message)
            panel.progress.set_progress(panel.progress.progress, label="Weight load failed")

        self.agent_weights_flash_until_ms = pygame.time.get_ticks() + 8000

    def new_game(self) -> None:
        self.session.reset()
        self.latest_analysis = None
        self.is_agent_thinking = False
        self.game_logged = False
        self.mcts.reset()

    def _trigger_agent_search(self) -> None:
        self.is_agent_thinking = True
        board = self.session.board.copy()
        history = [b.copy() for b in self.session.history]

        def worker() -> None:
            move, _, analysis = self.mcts.search(board, history, self.simulations, temperature=0.0)
            self.agent_move_queue.put((move, analysis))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_move(self, move: chess.Move) -> None:
        self.mcts.advance_root(move)
        self.session.push(move)
        if self.session.board.is_game_over():
            self._on_game_over()

    def _on_game_over(self) -> None:
        if self.game_logged:
            return
        self.game_logged = True
        board = self.session.board
        result = result_string(board)
        pgn_file = self.pgns_dir / f"human_match_{datetime.datetime.now():%Y%m%d_%H%M%S}.pgn"

        human_is_white = self.session.human_color == chess.WHITE
        game = chess.pgn.Game.from_board(board)
        game.headers["White"] = "Human" if human_is_white else self.agent_name
        game.headers["Black"] = self.agent_name if human_is_white else "Human"
        game.headers["Result"] = result
        pgn_file.write_text(f"{game}\n\n", encoding="utf-8")

        self.ledger.record_game(
            kind="human",
            white=game.headers["White"],
            black=game.headers["Black"],
            result=result,
            plies=len(board.move_stack),
            pgn_path=str(pgn_file),
            checkpoint_id=self.checkpoint_id,
        )

    # -- event loop ------------------------------------------------------

    def _poll_queues(self) -> None:
        self._poll_agent_reload()
        try:
            move, analysis = self.agent_move_queue.get_nowait()
            self.latest_analysis = analysis
            self.is_agent_thinking = False
            if move in self.session.board.legal_moves:
                self._apply_move(move)
        except queue.Empty:
            pass

        while True:
            try:
                ev = self.worker_queue.get_nowait()
            except queue.Empty:
                break
            if ev.get("type") == "train_viz":
                self.train_viz_state.handle_event(ev.get("data", {}))
                self._sync_train_live_btn()
                continue
            kind = ev.get("kind")
            if kind in self.panels:
                self.panels[kind].handle_worker_event(ev)

    def _activity_status(self) -> Tuple[str, bool]:
        """Return (message, is_busy) for the global status strip."""
        # Training tab visualiser has its own stage/progress UI; skip the global strip here.
        if self.current_tab == "train" and self.train_view_mode == "visualiser":
            if self.agent_weights_loading:
                return self.agent_weights_message, True
            if pygame.time.get_ticks() < self.agent_weights_flash_until_ms and self.agent_weights_message:
                return self.agent_weights_message, False
            return "", False

        if self.train_panel.worker.is_running():
            label = self.train_panel.progress.label or "Training in progress..."
            return label, True
        if self.bench_panel.worker.is_running():
            label = self.bench_panel.progress.label or "Benchmark in progress..."
            return label, True
        if self.agent_weights_loading:
            return self.agent_weights_message, True
        if self.is_agent_thinking:
            return "Agent thinking (MCTS search)...", True
        if pygame.time.get_ticks() < self.agent_weights_flash_until_ms:
            return self.agent_weights_message, False
        return "", False

    def _handle_play_event(self, event: pygame.event.Event) -> None:
        s = self.session
        if event.type not in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
            return
        if event.type == pygame.MOUSEMOTION:
            s.mouse_motion(event.pos)
            return
        if event.button != 1:
            return
        sq = self.renderer.pos_to_square(event.pos, s.flipped)
        if event.type == pygame.MOUSEBUTTONDOWN:
            move = s.mouse_down(event.pos, sq) if not self.is_agent_thinking else None
        else:
            move = s.mouse_up(sq)
        if move is not None:
            self._apply_move(move)

    def _draw(self) -> None:
        r, screen = self.renderer, self.screen
        screen.fill(COLOR_BG)
        self.tab_bar.render(screen, r.font_md)
        status_msg, status_busy = self._activity_status()
        if status_msg:
            r.draw_activity_strip(screen, status_msg, busy=status_busy)

        if self.current_tab == "play":
            s = self.session
            dragging = s.is_dragging
            r.draw_board(
                screen, s.board, s.flipped, s.selected_sq, s.last_move, s.legal_destinations, self.latest_analysis,
                drag_from_sq=s.drag_from_sq if dragging else None,
                drag_piece=s.drag_piece if dragging else None,
                drag_pos=s.drag_pos if dragging else None,
            )
            r.draw_analysis_panel(
                screen, s.board, self.latest_analysis, self.agent_name, self.is_agent_thinking,
                weights_status=self.agent_weights_message,
                weights_loading=self.agent_weights_loading,
            )
        elif self.current_tab == "train":
            champion = self.stats_summary.get("champion")
            train_status = self.train_panel.status
            if self.agent_weights_loading:
                train_status = "loading"
            r.draw_job_header(
                screen, "Training", train_status,
                f"Champion: {champion['name'] if champion else 'None'}",
                secondary=self.agent_weights_message if self.agent_weights_loading else "",
            )
            self.train_reset_btn.render(screen, r.font_sm)
            for sel in self.train_panel.selectors:
                sel.render(screen, r.font_md)
            self.train_panel.action.render(screen, r.font_md)
            self.train_view_btn.render(screen, r.font_md)

            if self.train_view_mode == "visualiser":
                self.train_pace_btn.render(screen, r.font_md)
                self.train_live_btn.render(screen, r.font_md)
                self.train_prev_btn.render(screen, r.font_md)
                self.train_next_btn.render(screen, r.font_md)
                self.train_end_btn.render(screen, r.font_md)
                self.train_visualizer.render(screen)
            else:
                self.train_panel.progress.render(screen, r.font_sm)
                self.train_panel.log.render(screen, r.font_sm)
        elif self.current_tab == "benchmark":
            elo = self.latest_bench_elo
            badge = f"Latest Elo: {elo['elo']:.0f} (CI: [{elo['ci_low']:.0f}, {elo['ci_high']:.0f}])" if elo else ""
            r.draw_job_header(screen, "Stockfish Gauntlet & Calibrated Elo", self.bench_panel.status, badge, badge=True)
            self.bench_panel.render(screen, r.font_md, r.font_sm)
        elif self.current_tab == "stats":
            r.draw_stats_tab(screen, self.stats_summary, self.stats_elo_history, self.stats_checkpoints)
            self.stats_refresh_btn.render(screen, r.font_sm)

    def run(self) -> None:
        running = True
        while running:
            self._poll_queues()

            s = self.session
            if self.current_tab == "play" and not s.board.is_game_over() and not self.is_agent_thinking \
                    and s.board.turn != s.human_color:
                self._trigger_agent_search()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                event = self._event_in_render_pixels(event)
                if self.tab_bar.handle_event(event):
                    continue
                if self.current_tab == "play":
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        running = False
                    self._handle_play_event(event)
                elif self.current_tab == "train":
                    self.train_reset_btn.handle_event(event)
                    for sel in self.train_panel.selectors:
                        sel.handle_event(event)
                    self.train_panel.action.handle_event(event)
                    self.train_view_btn.handle_event(event)
                    if self.train_view_mode == "visualiser":
                        self.train_pace_btn.handle_event(event)
                        self.train_live_btn.handle_event(event)
                        self.train_prev_btn.handle_event(event)
                        self.train_next_btn.handle_event(event)
                        self.train_end_btn.handle_event(event)
                        self.train_visualizer.handle_event(event)
                    else:
                        self.train_panel.log.handle_event(event)
                elif self.current_tab == "benchmark":
                    self.bench_panel.handle_event(event)
                elif self.current_tab == "stats":
                    self.stats_refresh_btn.handle_event(event)

            self._draw()
            self._present()
            self.clock.tick(60)

        if self._hidpi is not None:
            self._hidpi.close()
        pygame.quit()


def run_app(config_path: str = DEFAULT_CONFIG_PATH, checkpoint: str = "", color: str = "white") -> None:
    ChessApp(config_path, checkpoint, color).run()
