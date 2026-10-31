"""Comprehensive tests for the training visualiser: state reducer, telemetry events, and rendering."""

from __future__ import annotations

import os
import chess
import pygame

# Ensure headless pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"

from rl_chess.gui.train_viz import (
    TrainingVisualizer,
    TrainingVizState,
)
from rl_chess.mcts import MCTS
from rl_chess.model import create_model
from rl_chess.train.games import evaluate_candidate, play_selfplay_game


def test_viz_state_reducer_full_cycle():
    state = TrainingVizState()
    assert state.phase == "idle"

    # 1. gen_start
    state.handle_event({
        "type": "gen_start",
        "generation": 1,
        "gen_index": 1,
        "total_generations": 2,
        "champion_name": "Champion-Gen0",
        "champion_gen": 0,
        "games_per_gen": 5,
        "sims": 32,
        "temp_init": 1.0,
        "temp_final": 0.0,
        "temp_decay_plies": 30,
        "replay_buffer_size": 100,
    })
    assert state.phase == "selfplay"
    assert state.generation == 1
    assert state.sims == 32
    assert state.replay_buffer_size == 100

    # 2. game_start
    state.handle_event({
        "type": "game_start",
        "phase": "selfplay",
        "generation": 1,
        "game_idx": 1,
        "total_games": 5,
        "white": "Champion-Gen0",
        "black": "Champion-Gen0",
    })
    assert state.current_game is not None
    assert state.current_game.game_idx == 1
    assert state.current_game.white == "Champion-Gen0"

    # 3. moves
    b = chess.Board()
    m1 = chess.Move.from_uci("e2e4")
    b.push(m1)
    state.handle_event({
        "type": "move",
        "generation": 1,
        "game_idx": 1,
        "ply": 1,
        "san": "e4",
        "uci": "e2e4",
        "from_sq": chess.E2,
        "to_sq": chess.E4,
        "mover": "white",
        "fen": b.fen(),
        "temperature": 1.0,
        "add_noise": True,
        "sims": 32,
        "root_value": 0.15,
        "root_value_white": 0.15,
        "candidates": [
            {"san": "e4", "uci": "e2e4", "from_sq": chess.E2, "to_sq": chess.E4, "prior": 0.35, "visits": 20, "visit_share": 0.625, "q_value": 0.18},
            {"san": "d4", "uci": "d2d4", "from_sq": chess.D2, "to_sq": chess.D4, "prior": 0.30, "visits": 12, "visit_share": 0.375, "q_value": 0.12},
        ],
        "pv": ["e2e4", "e7e5"],
        "pv_san": "e4 e5",
        "entropy": 0.85,
    })

    assert len(state.current_game.moves) == 1
    assert state.get_viewing_ply() == 1
    board_view, move_view = state.get_viewing_board_and_move()
    assert move_view is not None
    assert move_view.san == "e4"
    assert board_view.piece_at(chess.E4) is not None

    # 4. game_end
    state.handle_event({
        "type": "game_end",
        "generation": 1,
        "game_idx": 1,
        "total_games": 5,
        "result": "1-0",
        "plies": 1,
        "termination_reason": "Resignation",
        "white_val": 1.0,
        "samples_count": 1,
        "final_fen": b.fen(),
        "replay_buffer_size": 101,
    })
    assert len(state.finished_games) == 1
    assert state.finished_games[0].result == "1-0"
    assert state.replay_buffer_size == 101

    # 5. train_step (optimization)
    state.handle_event({
        "type": "train_step",
        "generation": 1,
        "step": 1,
        "total_steps": 10,
        "loss_total": 2.45,
        "loss_policy": 1.85,
        "loss_value": 0.60,
        "lr": 0.001,
        "replay_buffer_size": 101,
    })
    assert state.phase == "training"
    assert len(state.loss_history) == 1
    assert state.loss_history[0].loss_total == 2.45

    # 6. arena_game
    state.handle_event({
        "type": "arena_game",
        "generation": 1,
        "game_idx": 1,
        "total_games": 2,
        "candidate_color": "white",
        "result": "1-0",
        "candidate_points": 1.0,
        "plies": 40,
        "running_score": 1.0,
        "threshold": 0.55,
    })
    assert state.phase == "arena"
    assert state.arena_candidate_wins == 1
    assert len(state.arena_games) == 1

    # 7. gen_end
    state.handle_event({
        "type": "gen_end",
        "generation": 1,
        "promoted": True,
        "win_rate": 0.75,
        "threshold": 0.55,
        "champion_name": "gen_1",
    })
    assert state.last_gen_summary is not None
    assert state.last_gen_summary["promoted"] is True
    assert state.champion_name == "gen_1"


def test_viz_state_navigation():
    state = TrainingVizState()
    state.handle_event({"type": "gen_start", "generation": 1})
    state.handle_event({"type": "game_start", "game_idx": 1})

    b = chess.Board()
    for ply_num, uci_move in enumerate(["e2e4", "e7e5", "g1f3"], start=1):
        m = chess.Move.from_uci(uci_move)
        b.push(m)
        state.handle_event({
            "type": "move",
            "ply": ply_num,
            "san": uci_move,
            "uci": uci_move,
            "from_sq": m.from_square,
            "to_sq": m.to_square,
            "fen": b.fen(),
            "root_value_white": 0.1 * ply_num,
            "candidates": [],
        })

    assert state.get_viewing_ply() == 3
    assert state.follow_live is True

    # Step back
    state.step_ply(-1)
    assert state.get_viewing_ply() == 2
    assert state.follow_live is False

    state.go_to_start()
    assert state.get_viewing_ply() == 0

    state.go_to_end()
    assert state.get_viewing_ply() == 3

    state.handle_event({"type": "game_end", "result": "1-0", "plies": 3, "final_fen": b.fen()})
    assert len(state.finished_games) == 1

    state.select_game(0)
    assert state.viewing_finished_idx == 0
    state.select_game(None)
    assert state.viewing_finished_idx is None
    assert state.follow_live is True

    # Pace cycling
    assert state.pace_mode == "Max"
    assert state.cycle_pace() == "0.2s"
    assert state.pace_delay_sec == 0.2
    assert state.cycle_pace() == "0.5s"
    assert state.cycle_pace() == "1.0s"
    assert state.cycle_pace() == "Max"
    assert state.pace_delay_sec == 0.0


def test_training_visualizer_headless_render():
    pygame.init()
    surface = pygame.Surface((1040, 620))
    state = TrainingVizState()
    viz = TrainingVisualizer(state, ui_scale=1.0)

    # 1. Render empty/idle state
    viz.render(surface)

    # 2. Render active self-play state
    state.handle_event({"type": "gen_start", "generation": 1, "sims": 64})
    state.handle_event({"type": "game_start", "phase": "selfplay", "game_idx": 1, "total_games": 10})

    b = chess.Board()
    m = chess.Move.from_uci("e2e4")
    b.push(m)
    state.handle_event({
        "type": "move",
        "ply": 1,
        "san": "e4",
        "uci": "e2e4",
        "from_sq": chess.E2,
        "to_sq": chess.E4,
        "fen": b.fen(),
        "temperature": 1.0,
        "add_noise": True,
        "sims": 64,
        "root_value": 0.2,
        "root_value_white": 0.2,
        "candidates": [
            {"san": "e4", "uci": "e2e4", "from_sq": chess.E2, "to_sq": chess.E4, "prior": 0.4, "visits": 40, "visit_share": 0.625, "q_value": 0.22},
        ],
        "pv": ["e2e4"],
        "pv_san": "e4",
        "entropy": 0.5,
    })
    viz.render(surface)

    # Test event handling: step left/right keys
    ev_left = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_LEFT})
    assert viz.handle_event(ev_left) is True
    assert state.get_viewing_ply() == 0

    ev_right = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHT})
    assert viz.handle_event(ev_right) is True
    assert state.get_viewing_ply() == 1

    # 3. Render training loss curve state
    state.handle_event({
        "type": "train_step",
        "step": 1,
        "total_steps": 10,
        "loss_total": 1.5,
        "loss_policy": 1.0,
        "loss_value": 0.5,
        "lr": 0.001,
        "replay_buffer_size": 200,
    })
    state.handle_event({
        "type": "train_step",
        "step": 2,
        "total_steps": 10,
        "loss_total": 1.3,
        "loss_policy": 0.9,
        "loss_value": 0.4,
        "lr": 0.001,
        "replay_buffer_size": 200,
    })
    viz.render(surface)

    # 4. Render arena gating state
    state.handle_event({
        "type": "arena_game",
        "game_idx": 1,
        "total_games": 5,
        "candidate_color": "white",
        "result": "1-0",
        "candidate_points": 1.0,
        "plies": 35,
        "running_score": 1.0,
        "threshold": 0.55,
    })
    state.handle_event({
        "type": "gen_end",
        "promoted": True,
        "win_rate": 0.8,
        "threshold": 0.55,
        "champion_name": "gen_1",
    })
    viz.render(surface)


def test_selfplay_and_arena_engine_callbacks():
    """Verify that play_selfplay_game and evaluate_candidate invoke telemetry callbacks."""
    net = create_model(num_blocks=2, channels=32, device="cpu")
    mcts = MCTS(net, device="cpu")

    moves_received = []
    game_ends_received = []

    samples, result, plies = play_selfplay_game(
        mcts,
        num_simulations=4,
        max_plies=2,
        on_move=lambda m: moves_received.append(m),
        on_game_end=lambda ge: game_ends_received.append(ge),
    )

    assert len(moves_received) >= 1
    assert "san" in moves_received[0]
    assert "root_value_white" in moves_received[0]
    assert "candidates" in moves_received[0]
    assert len(game_ends_received) == 1
    assert "termination_reason" in game_ends_received[0]

    # Test arena callbacks
    cand_net = create_model(num_blocks=2, channels=32, device="cpu")
    champ_net = create_model(num_blocks=2, channels=32, device="cpu")

    arena_starts = []
    arena_ends = []

    promoted, win_rate, records = evaluate_candidate(
        candidate_net=cand_net,
        champion_net=champ_net,
        num_games=1,
        num_simulations=4,
        on_game_start=lambda gs: arena_starts.append(gs),
        on_game_end=lambda ge: arena_ends.append(ge),
        device="cpu",
    )
    assert len(arena_starts) == 1
    assert arena_starts[0]["candidate_color"] == "white"
    assert arena_starts[0]["total_games"] == 1
    assert len(arena_ends) == 1
    assert len(records) == 1
    assert records[0]["candidate_color"] == "white"


def test_viz_state_arena_start_and_live_wld():
    state = TrainingVizState()
    state.handle_event({"type": "arena_start", "total_games": 20, "threshold": 0.55})
    assert state.phase == "arena"
    assert state.arena_total_games == 20
    state.handle_event({
        "type": "arena_game",
        "game_idx": 1,
        "total_games": 20,
        "candidate_color": "white",
        "result": "1-0",
        "candidate_points": 1.0,
        "plies": 40,
        "running_score": 1.0,
        "threshold": 0.55,
    })
    assert state.arena_candidate_wins == 1
    assert state.arena_candidate_losses == 0


def test_viz_state_arena_game_start_candidate_color():
    state = TrainingVizState()
    state.handle_event({
        "type": "game_start",
        "phase": "arena",
        "game_idx": 2,
        "total_games": 10,
        "candidate_color": "black",
        "white": "gen_2",
        "black": "gen_1",
    })
    assert state.current_game is not None
    assert state.current_game.phase == "arena"
    assert state.current_game.candidate_color == "black"
    assert state.current_game.white == "gen_2"
    assert state.current_game.black == "gen_1"
