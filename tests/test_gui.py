"""Tests for Pygame GUI widgets, multi-tab renderers, and headless application logic."""

from __future__ import annotations

import os
import queue
import chess
import pygame

# Run pygame in headless mode for CI/tests
os.environ["SDL_VIDEODRIVER"] = "dummy"

from rl_chess.gui.app import ChessApp
from rl_chess.gui.game_end import game_end_notice
from rl_chess.gui.renderer import ChessRenderer
from rl_chess.gui.widgets import (
    Button,
    CycleButton,
    LogBox,
    ProgressBar,
    TabBar,
)
from rl_chess.gui.play import PlaySession, resolve_human_move
from rl_chess.gui.workers import BackgroundWorker
from rl_chess.mcts import CandidateMoveAnalysis, MCTSAnalysis


def test_game_end_notice_messages():
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    notice = game_end_notice(board, human_color=chess.WHITE)
    assert notice.tone == "loss"
    assert notice.title == "You lose"
    assert notice.subtitle == "Checkmate"

    board2 = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    notice2 = game_end_notice(board2, human_color=chess.WHITE)
    assert notice2.tone == "win"
    assert notice2.title == "You win!"

    board3 = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    notice3 = game_end_notice(board3, human_color=chess.BLACK)
    assert notice3.tone == "loss"


def test_game_end_overlay_render():
    pygame.init()
    renderer = ChessRenderer(board_size=560)
    surface = pygame.Surface((1040, 620))
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    notice = game_end_notice(board, human_color=chess.WHITE)
    renderer.draw_board(surface, board)
    renderer.draw_game_end_overlay(surface, board, notice, elapsed_ms=600, flipped=False)
    assert surface.get_at((renderer.board_offset_x + 40, renderer.board_offset_y + 40))[3] == 255


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
    cycle_btn = CycleButton(
        rect=(10, 60, 120, 30),
        label="Mode",
        options=["Easy", "Medium", "Hard"],
        initial_index=0,
    )
    assert cycle_btn.value == "Easy"
    ev_cycle = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (20, 70)})
    cycle_btn.handle_event(ev_cycle)
    assert cycle_btn.value == "Medium"
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

    renderer.draw_job_header(surface, "Training", status="idle", info="Current Champion: gen_0")

    latest_elo = {"elo": 1250.0, "ci_low": 1180.0, "ci_high": 1320.0}
    renderer.draw_job_header(surface, "Benchmark", status="running", info="Latest Elo: 1250", badge=True)

    # Test Stats tab
    summary = {
        "total_games": 25,
        "selfplay_games": 20,
        "human_games": 3,
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
    app = ChessApp(config_path="configs/config.yaml")
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
    app.worker_queue.put({"kind": "train", "type": "log", "data": "Test log output"})
    app.worker_queue.put({
        "kind": "train", "type": "progress",
        "data": {"phase": "Self-Play", "generation": 1, "current": 1, "total": 10, "fraction": 0.1},
    })
    app._poll_queues()
    assert "Test log output" in app.train_panel.log.lines[-1]
    assert app.train_panel.progress.progress == 0.1


def test_high_resolution_mouse_coordinates_are_scaled():
    pygame.init()
    app = ChessApp(config_path="configs/config.yaml")

    class FakeWindow:
        def to_surface(self, pos):
            return (pos[0] * 2, pos[1] * 2)

        def surface_per_point(self):
            return 2.0

    app._hidpi = FakeWindow()

    event = pygame.event.Event(
        pygame.MOUSEMOTION,
        {"pos": (125, 75), "rel": (3, -2), "buttons": (0, 0, 0)},
    )
    scaled = app._event_in_render_pixels(event)

    assert scaled.pos == (250, 150)
    assert scaled.rel == (6, -4)


def test_background_worker_reports_events():
    q: queue.Queue = queue.Queue()

    def job(n, on_log, stop_event):
        on_log(f"ran {n}")

    worker = BackgroundWorker("train", job, q)
    assert worker.start(n=3)
    worker._thread.join(timeout=5)

    events = [q.get_nowait() for _ in range(q.qsize())]
    assert [(e["type"], e["data"]) for e in events] == [
        ("status", "running"), ("log", "ran 3"), ("status", "completed"),
    ]


def test_resolve_human_move_promotion_and_illegal():
    board = chess.Board("8/P6k/8/8/8/8/8/K7 w - - 0 1")
    assert resolve_human_move(board, chess.A7, chess.A8) == chess.Move(chess.A7, chess.A8, promotion=chess.QUEEN)
    assert resolve_human_move(board, chess.A7, chess.A7) is None
    assert resolve_human_move(board, chess.A1, chess.H8) is None


def test_play_session_click_and_drag():
    session = PlaySession(chess.WHITE, drag_threshold=6)

    # Click-to-move: select e2, then click e4
    assert session.mouse_down((0, 0), chess.E2) is None
    assert session.selected_sq == chess.E2 and chess.E4 in session.legal_destinations
    assert session.mouse_up(chess.E2) is None  # released in place keeps selection
    assert session.selected_sq == chess.E2
    move = session.mouse_down((0, 0), chess.E4)
    assert move == chess.Move.from_uci("e2e4")
    session.push(move)
    assert session.selected_sq is None and session.last_move == move

    # Not the human's turn now
    assert session.mouse_down((0, 0), chess.E7) is None
    session.push(chess.Move.from_uci("e7e5"))

    # Drag g1 -> f3
    session.mouse_down((10, 10), chess.G1)
    session.mouse_motion((40, 40))
    assert session.is_dragging
    assert session.mouse_up(chess.F3) == chess.Move.from_uci("g1f3")
    assert not session.is_dragging


def test_train_tab_visualizer_and_toggle():
    pygame.init()
    app = ChessApp(config_path="configs/config.yaml")

    # Switch to train tab
    app.tab_bar.active_tab = "train"
    app._on_tab_change("train")
    assert app.current_tab == "train"
    assert app.train_view_mode == "visualiser"

    # 1. Render in visualiser view (idle)
    app._draw()

    # 2. Feed telemetry events to queue
    app.worker_queue.put({
        "type": "train_viz",
        "data": {"type": "gen_start", "generation": 1, "champion_name": "gen_0", "sims": 64},
    })
    b = chess.Board()
    b.push_san("e4")
    app.worker_queue.put({
        "type": "train_viz",
        "data": {
            "type": "move",
            "generation": 1,
            "game_idx": 1,
            "ply": 1,
            "san": "e4",
            "uci": "e2e4",
            "from_sq": chess.E2,
            "to_sq": chess.E4,
            "fen": b.fen(),
            "root_value_white": 0.1,
            "candidates": [
                {"san": "e4", "uci": "e2e4", "from_sq": chess.E2, "to_sq": chess.E4, "prior": 0.5, "visits": 30, "visit_share": 0.6, "q_value": 0.15},
            ],
        },
    })
    app._poll_queues()
    assert app.train_viz_state.generation == 1
    assert app.train_viz_state.current_game is not None
    assert len(app.train_viz_state.current_game.moves) == 1

    # Render during selfplay
    app._draw()

    # Toggle to Log view
    app._toggle_train_view()
    assert app.train_view_mode == "log"
    assert app.train_view_btn.text == "View: Logs"
    app._draw()

    # Toggle back to Visualiser view
    app._toggle_train_view()
    assert app.train_view_mode == "visualiser"
    assert app.train_view_btn.text == "View: Board"
    app._draw()

    # Test pace toggle
    app._toggle_train_pace()
    assert app.train_viz_state.pace_mode == "0.2s"

    # Test live follow toggle
    app._toggle_train_live()
    assert app.train_viz_state.follow_live is False
    assert app.train_live_btn.text == "Paused"


def test_training_toolbar_controls_fit_without_overlap():
    pygame.init()
    app = ChessApp(config_path="configs/config.yaml")
    controls = [
        *app.train_panel.selectors,
        app.train_panel.action,
        app.train_view_btn,
        app.train_pace_btn,
        app.train_live_btn,
        app.train_prev_btn,
        app.train_next_btn,
        app.train_end_btn,
    ]

    for control in controls:
        assert control.rect.left >= 20
        assert control.rect.right <= 1020
    for index, control in enumerate(controls):
        assert all(not control.rect.colliderect(other.rect) for other in controls[index + 1:])
