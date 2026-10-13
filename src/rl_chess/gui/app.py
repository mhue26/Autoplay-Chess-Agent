"""Pygame application entry point for human vs agent play."""

from __future__ import annotations

import datetime
import queue
import threading
from pathlib import Path
from typing import List, Optional
import chess
import chess.pgn
import pygame
import torch
import yaml

from rl_chess.gui.renderer import ChessRenderer
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
        pygame.display.set_caption("RL-Chess Agent | AlphaZero & Live Analysis")

        self.board_size = 560
        self.panel_width = 360
        self.screen = pygame.display.set_mode((self.board_size + self.panel_width, self.board_size))
        self.clock = pygame.time.Clock()
        self.renderer = ChessRenderer(board_size=self.board_size, panel_width=self.panel_width)

        # Config & Ledger
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        paths = self.cfg.get("paths", {})
        self.ledger = Ledger(paths.get("ledger_db", "data/ledger.db"))
        self.pgns_dir = Path(paths.get("pgns_dir", "data/pgns"))
        self.pgns_dir.mkdir(parents=True, exist_ok=True)

        self.device = resolve_device(self.cfg.get("device", "auto"))
        self.simulations = self.cfg.get("mcts", {}).get("num_simulations", 64)

        # Agent model setup
        self.net = create_model(profile=self.cfg.get("profile", "tiny"), device=str(self.device))
        self.checkpoint_id: Optional[int] = None
        self.agent_name = "Random-Agent"

        self._load_agent_checkpoint(checkpoint_target)
        self.mcts = MCTS(self.net, device=str(self.device))

        # Game state
        self.human_color = chess.WHITE if human_color_str.lower() == "white" else chess.BLACK
        self.flipped = (self.human_color == chess.BLACK)
        self.board = chess.Board()
        self.history: List[chess.Board] = []
        self.last_move: Optional[chess.Move] = None
        self.selected_sq: Optional[chess.Square] = None
        self.legal_destinations: List[chess.Square] = []

        # Analysis & Threading
        self.latest_analysis: Optional[MCTSAnalysis] = None
        self.is_agent_thinking = False
        self.agent_queue: queue.Queue = queue.Queue()
        self.game_logged = False

    def _load_agent_checkpoint(self, target: str) -> None:
        cp_path = None
        if target:
            p = Path(target)
            if p.exists():
                cp_path = p
                self.agent_name = p.stem
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
            print("No checkpoint found; running agent with initial random weights.")

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
                temperature=0.0,  # Competitive play
                add_noise=False,
            )
            self.agent_queue.put((move, analysis))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def _handle_click(self, pos: Tuple[int, int]) -> None:
        if self.board.is_game_over() or self.is_agent_thinking:
            return

        sq = self.renderer.pos_to_square(pos, self.flipped)
        if sq is None:
            return

        # If already selected a square, try to play move
        if self.selected_sq is not None:
            # Check promotion
            move = chess.Move(self.selected_sq, sq)
            # Auto promote to queen
            if chess.Move(self.selected_sq, sq, promotion=chess.QUEEN) in self.board.legal_moves:
                move = chess.Move(self.selected_sq, sq, promotion=chess.QUEEN)

            if move in self.board.legal_moves:
                self._apply_move(move)
                self.selected_sq = None
                self.legal_destinations = []
                return

        # Select piece
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

        # Save PGN
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pgn_file = self.pgns_dir / f"human_match_{ts}.pgn"

        pgn_game = chess.pgn.Game.from_board(self.board)
        pgn_game.headers["White"] = "Human" if self.human_color == chess.WHITE else self.agent_name
        pgn_game.headers["Black"] = self.agent_name if self.human_color == chess.WHITE else "Human"
        pgn_game.headers["Result"] = res
        with open(pgn_file, "w", encoding="utf-8") as f:
            print(pgn_game, file=f, end="\n\n")

        # Log in Ledger
        self.ledger.record_game(
            kind="human",
            white=pgn_game.headers["White"],
            black=pgn_game.headers["Black"],
            result=res,
            plies=len(self.board.move_stack),
            pgn_path=str(pgn_file),
            checkpoint_id=self.checkpoint_id,
        )
        print(f"Human match complete: {res} saved to ledger & {pgn_file}")

    def run(self) -> None:
        running = True
        while running:
            # Check agent worker queue
            try:
                move, analysis = self.agent_queue.get_nowait()
                self.latest_analysis = analysis
                self.is_agent_thinking = False
                if move in self.board.legal_moves:
                    self._apply_move(move)
            except queue.Empty:
                pass

            # Trigger agent move if it's agent's turn
            if not self.board.is_game_over() and not self.is_agent_thinking:
                if self.board.turn != self.human_color:
                    self._trigger_agent_search()

            # Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)
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

            # Render frame
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

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


def run_app(config_path: str = "configs/tiny.yaml", checkpoint: str = "", color: str = "white") -> None:
    app = ChessApp(config_path=config_path, checkpoint_target=checkpoint, human_color_str=color)
    app.run()
