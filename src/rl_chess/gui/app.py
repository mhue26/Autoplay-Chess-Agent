"""Unified Pygame application for Play, Training, Benchmarking, and Stats."""

from __future__ import annotations

import datetime
import queue
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import chess
import chess.pgn
import pygame
import torch
import yaml

from rl_chess.gui.renderer import ChessRenderer
from rl_chess.gui.widgets import (
    Button,
    COLOR_BG,
    COLOR_CARD,
    COLOR_DANGER,
    COLOR_DANGER_HOVER,
    COLOR_PRIMARY,
    COLOR_PRIMARY_HOVER,
    CycleButton,
    LogBox,
    ProgressBar,
    TabBar,
)
from rl_chess.gui.workers import BenchmarkWorker, TrainingWorker
from rl_chess.ledger.database import Ledger
from rl_chess.mcts.search import MCTS, MCTSAnalysis
from rl_chess.net.model import create_model, resolve_device


class ChessApp:
    def __init__(
        self,
        config_path: str = "configs/tiny.yaml",
        checkpoint_target: str = "",
        human_color_str: str = "white",
    ):
        pygame.init()
        pygame.display.set_caption("RL-Chess | Unified AlphaZero Dashboard")

        self.window_width = 1040
        self.window_height = 620
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.clock = pygame.time.Clock()

        # Renderer
        self.renderer = ChessRenderer(
            board_size=560,
            board_offset_x=20,
            board_offset_y=50,
            panel_x=600,
            panel_width=420,
        )

        # Config & Ledger
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        paths = self.cfg.get("paths", {})
        self.ledger_db_path = paths.get("ledger_db", "data/ledger.db")
        self.ledger = Ledger(self.ledger_db_path)
        self.pgns_dir = Path(paths.get("pgns_dir", "data/pgns"))
        self.pgns_dir.mkdir(parents=True, exist_ok=True)

        self.device = resolve_device(self.cfg.get("device", "auto"))
        self.simulations = self.cfg.get("mcts", {}).get("num_simulations", 64)

        # Top Navigation Tab Bar
        self.tab_bar = TabBar(
            rect=(0, 0, self.window_width, 44),
            tabs=[
                ("play", "Play vs Agent"),
                ("train", "Train Self-Play"),
                ("benchmark", "Benchmark Elo"),
                ("stats", "Stats & Ledger"),
            ],
            initial_tab="play",
            on_tab_change=self._on_tab_change,
        )
        self.current_tab = "play"

        # Worker Queue & Workers
        self.worker_queue: queue.Queue = queue.Queue()
        self.train_worker = TrainingWorker(self.worker_queue)
        self.bench_worker = BenchmarkWorker(self.worker_queue)

        # -------------------------------------------------------------
        # Tab 1: Play state
        # -------------------------------------------------------------
        self.net = create_model(profile=self.cfg.get("profile", "tiny"), device=str(self.device))
        self.checkpoint_id: Optional[int] = None
        self.agent_name = "Random-Agent"
        self._load_agent_checkpoint(checkpoint_target)
        self.mcts = MCTS(self.net, device=str(self.device))

        self.human_color = chess.WHITE if human_color_str.lower() == "white" else chess.BLACK
        self.flipped = (self.human_color == chess.BLACK)
        self.board = chess.Board()
        self.history: List[chess.Board] = []
        self.last_move: Optional[chess.Move] = None
        self.selected_sq: Optional[chess.Square] = None
        self.legal_destinations: List[chess.Square] = []

        self.latest_analysis: Optional[MCTSAnalysis] = None
        self.is_agent_thinking = False
        self.agent_move_queue: queue.Queue = queue.Queue()
        self.game_logged = False

        # -------------------------------------------------------------
        # Tab 2: Training Widgets
        # -------------------------------------------------------------
        self.train_profile_btn = CycleButton(
            rect=(20, 92, 180, 36),
            label="Profile",
            options=["tiny", "small", "medium"],
            initial_index=0,
            on_change=self._on_train_profile_change,
        )
        self.train_gens_btn = CycleButton(
            rect=(215, 92, 170, 36),
            label="Generations",
            options=["1 gen", "2 gens", "5 gens", "10 gens"],
            initial_index=0,
        )
        self.train_action_btn = Button(
            rect=(400, 92, 160, 36),
            text="Start Training",
            color=COLOR_PRIMARY,
            hover_color=COLOR_PRIMARY_HOVER,
            on_click=self._toggle_training,
        )
        self.train_progress_bar = ProgressBar(
            rect=(20, 140, 1000, 24),
            progress=0.0,
            label="Ready to train",
        )
        self.train_log_box = LogBox(
            rect=(20, 176, 1000, 424),
            max_lines=600,
        )
        self.train_status = "idle"

        # -------------------------------------------------------------
        # Tab 3: Benchmark Widgets
        # -------------------------------------------------------------
        self.bench_checkpoint_btn = CycleButton(
            rect=(20, 92, 220, 36),
            label="Checkpoint",
            options=self._get_checkpoint_options(),
            initial_index=0,
        )
        self.bench_games_btn = CycleButton(
            rect=(255, 92, 170, 36),
            label="Games / Rung",
            options=["5 games", "10 games", "20 games"],
            initial_index=1,
        )
        self.bench_action_btn = Button(
            rect=(440, 92, 160, 36),
            text="Run Gauntlet",
            color=COLOR_PRIMARY,
            hover_color=COLOR_PRIMARY_HOVER,
            on_click=self._toggle_benchmark,
        )
        self.bench_progress_bar = ProgressBar(
            rect=(20, 140, 1000, 24),
            progress=0.0,
            label="Ready to benchmark",
        )
        self.bench_log_box = LogBox(
            rect=(20, 176, 1000, 424),
            max_lines=600,
        )
        self.bench_status = "idle"
        self.latest_bench_elo: Optional[Dict[str, Any]] = None

        # -------------------------------------------------------------
        # Tab 4: Stats Cache
        # -------------------------------------------------------------
        self.stats_summary: Dict[str, Any] = {}
        self.stats_elo_history: List[Dict[str, Any]] = []
        self.stats_checkpoints: List[Any] = []
        self.stats_refresh_btn = Button(
            rect=(880, 54, 120, 32),
            text="Refresh Stats",
            on_click=self._refresh_stats,
        )
        self._refresh_stats()

    def _get_checkpoint_options(self) -> List[str]:
        cps = self.ledger.list_checkpoints()
        names = ["Champion (Latest)"]
        for c in cps:
            names.append(c.name)
        return names

    def _on_tab_change(self, tab_id: str) -> None:
        self.current_tab = tab_id
        if tab_id == "stats":
            self._refresh_stats()
        elif tab_id == "benchmark":
            self.bench_checkpoint_btn.set_options(self._get_checkpoint_options())

    def _on_train_profile_change(self, profile: str) -> None:
        cfg_path = f"configs/{profile}.yaml"
        if Path(cfg_path).exists():
            self.config_path = cfg_path
            with open(cfg_path, "r") as f:
                self.cfg = yaml.safe_load(f)
            self.train_log_box.append(f"Selected profile configuration: {cfg_path}")

    def _refresh_stats(self) -> None:
        try:
            self.stats_summary = self.ledger.get_stats_summary()
            self.stats_elo_history = self.ledger.get_elo_history()
            self.stats_checkpoints = self.ledger.list_checkpoints()
            if self.stats_summary.get("latest_elo"):
                self.latest_bench_elo = self.stats_summary["latest_elo"]
        except Exception as e:
            print(f"Error refreshing stats: {e}")

    def _load_agent_checkpoint(self, target: str) -> None:
        cp_path = None
        if target and target != "Champion (Latest)":
            p = Path(target)
            if p.exists():
                cp_path = p
                self.agent_name = p.stem
            else:
                rec = next((c for c in self.ledger.list_checkpoints() if c.name == target), None)
                if rec and Path(rec.path).exists():
                    cp_path = Path(rec.path)
                    self.agent_name = f"{rec.name} (Gen {rec.generation})"
                    self.checkpoint_id = rec.id
        else:
            champ = self.ledger.get_champion_checkpoint()
            if champ and Path(champ.path).exists():
                cp_path = Path(champ.path)
                self.agent_name = f"{champ.name} (Gen {champ.generation})"
                self.checkpoint_id = champ.id

        if cp_path and cp_path.exists():
            print(f"Loading agent weights from {cp_path}...")
            state = torch.load(cp_path, map_location=self.device)
            self.net.load_state_dict(state)
            self.net.eval()
        else:
            print("No checkpoint loaded; running agent with initial random weights.")

    def new_game(self) -> None:
        self.board = chess.Board()
        self.history = []
        self.last_move = None
        self.selected_sq = None
        self.legal_destinations = []
        self.latest_analysis = None
        self.is_agent_thinking = False
        self.game_logged = False
        self.mcts.reset()

    # -------------------------------------------------------------
    # Training Worker Trigger
    # -------------------------------------------------------------
    def _toggle_training(self) -> None:
        if self.train_worker.is_running():
            self.train_worker.request_stop()
            self.train_action_btn.text = "Stopping..."
            self.train_action_btn.is_disabled = True
        else:
            gen_str = self.train_gens_btn.value.split()[0]
            generations = int(gen_str)
            profile = self.train_profile_btn.value
            cfg_path = f"configs/{profile}.yaml"

            self.train_log_box.append(f"Starting training run ({generations} gens, config={cfg_path})...")
            started = self.train_worker.start(config_path=cfg_path, generations=generations)
            if started:
                self.train_status = "running"
                self.train_action_btn.text = "Stop Training"
                self.train_action_btn.color = COLOR_DANGER
                self.train_action_btn.hover_color = COLOR_DANGER_HOVER

    # -------------------------------------------------------------
    # Benchmark Worker Trigger
    # -------------------------------------------------------------
    def _toggle_benchmark(self) -> None:
        if self.bench_worker.is_running():
            self.bench_worker.request_stop()
            self.bench_action_btn.text = "Stopping..."
            self.bench_action_btn.is_disabled = True
        else:
            games_str = self.bench_games_btn.value.split()[0]
            games_count = int(games_str)
            cp_choice = self.bench_checkpoint_btn.value
            target = "" if "Champion" in cp_choice else cp_choice
            profile = self.train_profile_btn.value
            cfg_path = f"configs/{profile}.yaml"

            self.bench_log_box.append(f"Starting Stockfish benchmark ({games_count} games/rung)...")
            started = self.bench_worker.start(
                config_path=cfg_path,
                checkpoint_target=target,
                games_per_opponent=games_count,
            )
            if started:
                self.bench_status = "running"
                self.bench_action_btn.text = "Stop Gauntlet"
                self.bench_action_btn.color = COLOR_DANGER
                self.bench_action_btn.hover_color = COLOR_DANGER_HOVER

    # -------------------------------------------------------------
    # Agent Search in Play Tab
    # -------------------------------------------------------------
    def _trigger_agent_search(self) -> None:
        if self.is_agent_thinking or self.board.is_game_over():
            return

        self.is_agent_thinking = True
        board_copy = self.board.copy()
        history_copy = [b.copy() for b in self.history]

        def worker():
            move, _, analysis = self.mcts.search(
                board=board_copy,
                history=history_copy,
                num_simulations=self.simulations,
                temperature=0.0,
                add_noise=False,
            )
            self.agent_move_queue.put((move, analysis))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def _handle_board_click(self, pos: Tuple[int, int]) -> None:
        if self.board.is_game_over() or self.is_agent_thinking:
            return

        sq = self.renderer.pos_to_square(pos, self.flipped)
        if sq is None:
            return

        if self.selected_sq is not None:
            move = chess.Move(self.selected_sq, sq)
            if chess.Move(self.selected_sq, sq, promotion=chess.QUEEN) in self.board.legal_moves:
                move = chess.Move(self.selected_sq, sq, promotion=chess.QUEEN)

            if move in self.board.legal_moves:
                self._apply_move(move)
                self.selected_sq = None
                self.legal_destinations = []
                return

        piece = self.board.piece_at(sq)
        if piece and piece.color == self.board.turn and piece.color == self.human_color:
            self.selected_sq = sq
            self.legal_destinations = [
                m.to_square for m in self.board.legal_moves if m.from_square == sq
            ]
        else:
            self.selected_sq = None
            self.legal_destinations = []

    def _apply_move(self, move: chess.Move) -> None:
        self.history.append(self.board.copy())
        if len(self.history) > 8:
            self.history.pop(0)

        self.mcts.advance_root(move)
        self.board.push(move)
        self.last_move = move

        if self.board.is_game_over():
            self._on_game_over()

    def _on_game_over(self) -> None:
        if self.game_logged:
            return

        self.game_logged = True
        outcome = self.board.outcome(claim_draw=True)
        res = "1/2-1/2"
        if outcome and outcome.winner is not None:
            res = "1-0" if outcome.winner == chess.WHITE else "0-1"

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pgn_file = self.pgns_dir / f"human_match_{ts}.pgn"

        pgn_game = chess.pgn.Game.from_board(self.board)
        pgn_game.headers["White"] = "Human" if self.human_color == chess.WHITE else self.agent_name
        pgn_game.headers["Black"] = self.agent_name if self.human_color == chess.WHITE else "Human"
        pgn_game.headers["Result"] = res
        with open(pgn_file, "w", encoding="utf-8") as f:
            print(pgn_game, file=f, end="\n\n")

        self.ledger.record_game(
            kind="human",
            white=pgn_game.headers["White"],
            black=pgn_game.headers["Black"],
            result=res,
            plies=len(self.board.move_stack),
            pgn_path=str(pgn_file),
            checkpoint_id=self.checkpoint_id,
        )

    # -------------------------------------------------------------
    # Queue Poll & Event Loop
    # -------------------------------------------------------------
    def _poll_queues(self) -> None:
        # 1. Agent moves in Play tab
        try:
            move, analysis = self.agent_move_queue.get_nowait()
            self.latest_analysis = analysis
            self.is_agent_thinking = False
            if move in self.board.legal_moves:
                self._apply_move(move)
        except queue.Empty:
            pass

        # 2. Worker events (train & benchmark)
        while True:
            try:
                ev = self.worker_queue.get_nowait()
            except queue.Empty:
                break

            ev_type = ev.get("type")
            if ev_type == "log":
                txt = ev.get("text", "")
                target = ev.get("target")
                if target == "train" or target is None:
                    self.train_log_box.append(txt)
                if target == "bench" or target is None:
                    self.bench_log_box.append(txt)

            elif ev_type == "train_progress":
                d = ev.get("data", {})
                phase = d.get("phase", "")
                gen = d.get("generation", 0)
                cur = d.get("current", 0)
                tot = d.get("total", 0)
                frac = d.get("fraction", 0.0)
                lbl = f"Gen {gen}: {phase} ({cur}/{tot})"
                self.train_progress_bar.set_progress(frac, label=lbl)

            elif ev_type == "train_status":
                st = ev.get("status", "idle")
                self.train_status = st
                self.train_action_btn.is_disabled = False
                if st in ("completed", "stopped", "error"):
                    self.train_action_btn.text = "Start Training"
                    self.train_action_btn.color = COLOR_PRIMARY
                    self.train_action_btn.hover_color = COLOR_PRIMARY_HOVER
                    self._refresh_stats()
                    self._load_agent_checkpoint("")  # reload latest champion

            elif ev_type == "bench_progress":
                d = ev.get("data", {})
                comp = d.get("completed", 0)
                tot = d.get("total", 0)
                frac = d.get("fraction", 0.0)
                rung = d.get("rung", "")
                w, dr, l = d.get("wins", 0), d.get("draws", 0), d.get("losses", 0)
                lbl = f"Testing vs {rung} ({comp}/{tot}) - {w}W {dr}D {l}L"
                self.bench_progress_bar.set_progress(frac, label=lbl)

            elif ev_type == "bench_result":
                res = ev.get("data", {})
                self.latest_bench_elo = res
                self._refresh_stats()

            elif ev_type == "bench_status":
                st = ev.get("status", "idle")
                self.bench_status = st
                self.bench_action_btn.is_disabled = False
                if st in ("completed", "stopped", "error"):
                    self.bench_action_btn.text = "Run Gauntlet"
                    self.bench_action_btn.color = COLOR_PRIMARY
                    self.bench_action_btn.hover_color = COLOR_PRIMARY_HOVER

    def run(self) -> None:
        running = True
        while running:
            self._poll_queues()

            # Trigger agent move if play tab active and it's agent's turn
            if self.current_tab == "play":
                if not self.board.is_game_over() and not self.is_agent_thinking:
                    if self.board.turn != self.human_color:
                        self._trigger_agent_search()

            # Handle Pygame Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                # Tab bar always handles events
                new_tab = self.tab_bar.handle_event(event)
                if new_tab:
                    continue

                if self.current_tab == "play":
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        self._handle_board_click(event.pos)
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_n:
                            self.new_game()
                        elif event.key == pygame.K_f:
                            self.flipped = not self.flipped
                        elif event.key == pygame.K_u:
                            if len(self.board.move_stack) >= 2 and not self.is_agent_thinking:
                                self.board.pop()
                                self.board.pop()
                                self.last_move = None
                                self.mcts.reset()

                elif self.current_tab == "train":
                    self.train_profile_btn.handle_event(event)
                    self.train_gens_btn.handle_event(event)
                    self.train_action_btn.handle_event(event)
                    self.train_log_box.handle_event(event)

                elif self.current_tab == "benchmark":
                    self.bench_checkpoint_btn.handle_event(event)
                    self.bench_games_btn.handle_event(event)
                    self.bench_action_btn.handle_event(event)
                    self.bench_log_box.handle_event(event)

                elif self.current_tab == "stats":
                    self.stats_refresh_btn.handle_event(event)

            # -------------------------------------------------------------
            # Render Current Frame
            # -------------------------------------------------------------
            self.screen.fill(COLOR_BG)

            # Draw Navigation Tab Bar
            self.tab_bar.render(self.screen, self.renderer.font_md)

            # Draw Active Tab Content
            if self.current_tab == "play":
                self.renderer.draw_board(
                    surface=self.screen,
                    board=self.board,
                    flipped=self.flipped,
                    selected_sq=self.selected_sq,
                    last_move=self.last_move,
                    legal_destinations=self.legal_destinations,
                    analysis=self.latest_analysis,
                )
                self.renderer.draw_analysis_panel(
                    surface=self.screen,
                    board=self.board,
                    analysis=self.latest_analysis,
                    agent_name=self.agent_name,
                    is_thinking=self.is_agent_thinking,
                )

            elif self.current_tab == "train":
                champ_name = self.stats_summary.get("champion", {}).get("name") if self.stats_summary.get("champion") else "gen_0"
                self.renderer.draw_train_tab_header(
                    surface=self.screen,
                    status=self.train_status,
                    champion_name=champ_name or "None",
                )
                self.train_profile_btn.render(self.screen, self.renderer.font_md)
                self.train_gens_btn.render(self.screen, self.renderer.font_md)
                self.train_action_btn.render(self.screen, self.renderer.font_md)
                self.train_progress_bar.render(self.screen, self.renderer.font_sm)
                self.train_log_box.render(self.screen, self.renderer.font_sm)

            elif self.current_tab == "benchmark":
                cp_choice = self.bench_checkpoint_btn.value
                self.renderer.draw_benchmark_tab_header(
                    surface=self.screen,
                    status=self.bench_status,
                    checkpoint_name=cp_choice,
                    latest_elo=self.latest_bench_elo,
                )
                self.bench_checkpoint_btn.render(self.screen, self.renderer.font_md)
                self.bench_games_btn.render(self.screen, self.renderer.font_md)
                self.bench_action_btn.render(self.screen, self.renderer.font_md)
                self.bench_progress_bar.render(self.screen, self.renderer.font_sm)
                self.bench_log_box.render(self.screen, self.renderer.font_sm)

            elif self.current_tab == "stats":
                self.renderer.draw_stats_tab(
                    surface=self.screen,
                    summary=self.stats_summary,
                    elo_history=self.stats_elo_history,
                    checkpoints=self.stats_checkpoints,
                )
                self.stats_refresh_btn.render(self.screen, self.renderer.font_sm)

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


def run_app(config_path: str = "configs/tiny.yaml", checkpoint: str = "", color: str = "white") -> None:
    app = ChessApp(config_path=config_path, checkpoint_target=checkpoint, human_color_str=color)
    app.run()
