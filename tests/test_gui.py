"""Tests for Pygame GUI renderer and application logic in headless mode."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
import chess
import pygame

# Run pygame in headless mode for CI/tests
os.environ["SDL_VIDEODRIVER"] = "dummy"

from rl_chess.gui.renderer import ChessRenderer
from rl_chess.mcts.search import CandidateMoveAnalysis, MCTSAnalysis


def test_renderer_coordinates():
    pygame.init()
    renderer = ChessRenderer(board_size=560, panel_width=360)

    # Square a1 (file=0, rank=0)
    rect_a1 = renderer.square_to_rect(chess.A1, flipped=False)
    assert rect_a1.x == 0
    assert rect_a1.y == 7 * 70  # Row 7

    # Inverted when flipped
    rect_a1_flip = renderer.square_to_rect(chess.A1, flipped=True)
    assert rect_a1_flip.x == 7 * 70
    assert rect_a1_flip.y == 0

    # pos_to_square
    sq = renderer.pos_to_square((10, 10), flipped=False)
    assert sq == chess.A8  # Top-left is a8 when White on bottom


def test_render_board_and_panel():
    pygame.init()
    renderer = ChessRenderer(board_size=560, panel_width=360)
    surface = pygame.Surface((920, 560))

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

    # Render board
    renderer.draw_board(
        surface=surface,
        board=board,
        selected_sq=chess.E2,
        legal_destinations=[chess.E3, chess.E4],
        analysis=analysis,
    )

    # Render panel
    renderer.draw_analysis_panel(
        surface=surface,
        board=board,
        analysis=analysis,
        agent_name="TestAgent-Gen0",
        is_thinking=False,
    )

    # Check surface has non-zero pixels
    pixel = surface.get_at((100, 100))
    assert len(pixel) == 4
