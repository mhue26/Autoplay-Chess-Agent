"""Self-play game generation using AlphaZero MCTS."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
import chess
import chess.pgn
import numpy as np

from rl_chess.core.encoding import encode_board
from rl_chess.mcts.search import MCTS


def play_selfplay_game(
    mcts: MCTS,
    num_simulations: int = 64,
    temp_init: float = 1.0,
    temp_final: float = 0.0,
    temp_decay_plies: int = 30,
    max_plies: int = 200,
    pgn_path: Optional[str | Path] = None,
    white_name: str = "AlphaZero",
    black_name: str = "AlphaZero",
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float]], str, int]:
    """Execute a single self-play game and collect training trajectories.
    
    Returns:
        samples: List of (encoded_state, visit_distribution_pi, z_outcome)
        result: PGN result string ('1-0', '0-1', '1/2-1/2')
        plies: Total plies played
    """
    board = chess.Board()
    history: List[chess.Board] = []
    game_history: List[Tuple[np.ndarray, np.ndarray, chess.Color]] = []
    mcts.reset()

    # Track PGN game
    pgn_game = chess.pgn.Game()
    pgn_game.headers["Event"] = "Self-Play RL Training"
    pgn_game.headers["White"] = white_name
    pgn_game.headers["Black"] = black_name
    pgn_node = pgn_game

    ply = 0
    while not board.is_game_over() and ply < max_plies:
        temp = temp_init if ply < temp_decay_plies else temp_final
        add_noise = (ply < temp_decay_plies)

        # Store encoded state before move
        state_planes = encode_board(board, history)

        # MCTS search
        move, pi, _ = mcts.search(
            board=board,
            history=history,
            num_simulations=num_simulations,
            temperature=temp,
            add_noise=add_noise,
        )

        if move == chess.Move.null() or move not in board.legal_moves:
            break

        game_history.append((state_planes, pi, board.turn))

        # Advance board & history
        history.append(board.copy())
        if len(history) > 8:
            history.pop(0)

        pgn_node = pgn_node.add_variation(move)
        mcts.advance_root(move)
        board.push(move)
        ply += 1

    # Determine game outcome
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        result_str = "1/2-1/2"
        white_val = 0.0
    elif outcome.winner == chess.WHITE:
        result_str = "1-0"
        white_val = 1.0
    else:
        result_str = "0-1"
        white_val = -1.0

    pgn_game.headers["Result"] = result_str

    if pgn_path:
        path = Path(pgn_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            print(pgn_game, file=f, end="\n\n")

    # Map trajectory to target values from perspective of current player
    samples: List[Tuple[np.ndarray, np.ndarray, float]] = []
    for state, pi, turn in game_history:
        target_v = white_val if turn == chess.WHITE else -white_val
        samples.append((state, pi, target_v))

    return samples, result_str, ply
