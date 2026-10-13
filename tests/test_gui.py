"""Tests for Pygame GUI widgets, multi-tab renderers, and headless application logic."""

from __future__ import annotations

import os
import queue
import tempfile
from pathlib import Path
import chess
import pygame

# Run pygame in headless mode for CI/tests
os.environ["SDL_VIDEODRIVER"] = "dummy"

from rl_chess.gui.app import ChessApp
from rl_chess.gui.renderer import ChessRenderer
from rl_chess.gui.widgets import (
    Button,
    CycleButton,
    LogBox,
    ProgressBar,
    TabBar,
)
from rl_chess.gui.workers import BenchmarkWorker, TrainingWorker
from rl_chess.mcts.search import CandidateMoveAnalysis, MCTSAnalysis


def test_renderer_coordinates():
    pygame.init()
    renderer = ChessRenderer(
        board_size=560,
        board_offset_x=20,
        board_offset_y=50,
        panel_x=600,
        panel_width=420,
    )

    # Square a1 (file=0, rank=0)
    rect_a1 = renderer.square_to_rect(chess.A1, flipped=False)
    assert rect_a1.x == 20
    assert rect_a1.y == 50 + 7 * 70  # Row 7

    # Inverted when flipped
    rect_a1_flip = renderer.square_to_rect(chess.A1, flipped=True)
    assert rect_a1_flip.x == 20 + 7 * 70
    assert rect_a1_flip.y == 50

    # pos_to_square within board
    sq = renderer.pos_to_square((30, 60), flipped=False)
    assert sq == chess.A8  # Top-left square is a8

    # pos_to_square outside board
    assert renderer.pos_to_square((5, 5), flipped=False) is None
    assert renderer.pos_to_square((800, 300), flipped=False) is None


def test_render_board_and_panel():
    pygame.init()
    renderer = ChessRenderer(board_size=560)
    surface = pygame.Surface((1040, 620))

    board = chess.Board()
    analysis = MCTSAnalysis(
        root_value=0.25,
        total_simulations=64,
        candidates=[
            CandidateMoveAnalysis(
                action=0,
                move=chess.Move.from_uci("e2e4"),
                san="e4",
                visits=40,
                visit_share=0.625,
                prior=0.3,
                q_value=0.25,
            )
        ],
        principal_variation=[chess.Move.from_uci("e2e4"), chess.Move.from_uci("e7e5")],
        pv_san="1. e4 e5",
    )

    renderer.draw_board(
        surface=surface,
        board=board,
        selected_sq=chess.E2,
        legal_destinations=[chess.E3, chess.E4],
        analysis=analysis,
    )

    renderer.draw_analysis_panel(
        surface=surface,
        board=board,
        analysis=analysis,
        agent_name="TestAgent-Gen0",
        is_thinking=False,
    )

    pixel = surface.get_at((100, 100))
    assert len(pixel) == 4


def test_widget_primitives():
    pygame.init()
    surface = pygame.Surface((400, 400))
    font = pygame.font.SysFont("Arial", 14)

    # 1. Button
    clicked = []
    btn = Button(
        rect=(10, 10, 100, 30),
        text="ClickMe",
        on_click=lambda: clicked.append(True),
    )
    # Simulate click event
    ev_down = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (20, 20)})
    btn.handle_event(ev_down)
    assert len(clicked) == 1
    btn.render(surface, font)

    # 2. TabBar
    tab_changed = []
    tab_bar = TabBar(
        rect=(0, 0, 400, 40),
        tabs=[("tab1", "Tab One"), ("tab2", "Tab Two")],
        initial_tab="tab1",
        on_tab_change=lambda t: tab_changed.append(t),
    )
    assert tab_bar.active_tab == "tab1"
    ev_tab = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (250, 20)})
    tab_bar.handle_event(ev_tab)
    assert tab_bar.active_tab == "tab2"
    assert tab_changed == ["tab2"]
    tab_bar.render(surface, font)

    # 3. CycleButton
    cycle_val = []
    cycle_btn = CycleButton(
        rect=(10, 60, 120, 30),
        label="Mode",
        options=["Easy", "Medium", "Hard"],
        initial_index=0,
        on_change=lambda v: cycle_val.append(v),
    )
    assert cycle_btn.value == "Easy"
    ev_cycle = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (20, 70)})
    cycle_btn.handle_event(ev_cycle)
    assert cycle_btn.value == "Medium"
    assert cycle_val == ["Medium"]
    cycle_btn.render(surface, font)

    # 4. ProgressBar
    pb = ProgressBar(rect=(10, 110, 200, 20), progress=0.4, label="40% Complete")
    assert pb.progress == 0.4
    pb.set_progress(0.8, label="80% Done")
    assert pb.progress == 0.8
    pb.render(surface, font)

    # 5. LogBox
    log_box = LogBox(rect=(10, 140, 300, 150), max_lines=10)
    for i in range(15):
        log_box.append(f"Log message {i}")
    assert len(log_box.lines) == 10
    assert "Log message 14" in log_box.lines[-1]
    log_box.render(surface, font)


def test_multi_tab_renderers():
    pygame.init()
    renderer = ChessRenderer(board_size=560)
    surface = pygame.Surface((1040, 620))

    # Test Train tab header
    renderer.draw_train_tab_header(surface, status="idle", champion_name="gen_0")

    # Test Benchmark tab header with Elo
    latest_elo = {"elo": 1250.0, "ci_low": 1180.0, "ci_high": 1320.0}
    renderer.draw_benchmark_tab_header(surface, status="running", checkpoint_name="gen_1", latest_elo=latest_elo)

    # Test Stats tab
    summary = {
        "total_games": 25,
        "selfplay_games": 20,
        "selfplay_positions_estimated": 840,
        "training_steps": 150,
        "checkpoints_count": 2,
        "champion": {"name": "gen_1", "generation": 1},
        "latest_elo": latest_elo,
    }
    elo_hist = [
        {"checkpoint_name": "gen_1", "generation": 1, "elo": 1250.0, "ci_low": 1180.0, "ci_high": 1320.0, "n_games": 20, "created_at": "2026-10-13"}
    ]
    renderer.draw_stats_tab(surface, summary, elo_hist, [])


def test_chess_app_headless_cycle():
    pygame.init()
    app = ChessApp(config_path="configs/tiny.yaml")
    assert app.current_tab == "play"

    # Switch tabs
    app.tab_bar.active_tab = "train"
    app._on_tab_change("train")
    assert app.current_tab == "train"

    app.tab_bar.active_tab = "benchmark"
    app._on_tab_change("benchmark")
    assert app.current_tab == "benchmark"

    app.tab_bar.active_tab = "stats"
    app._on_tab_change("stats")
    assert app.current_tab == "stats"

    # Queue polling test
    app.worker_queue.put({"type": "log", "text": "Test log output"})
    app.worker_queue.put({"type": "train_progress", "data": {"phase": "Self-Play", "generation": 1, "current": 1, "total": 10, "fraction": 0.1}})
    app._poll_queues()

    assert "Test log output" in app.train_log_box.lines[-1]
    assert app.train_progress_bar.progress == 0.1
