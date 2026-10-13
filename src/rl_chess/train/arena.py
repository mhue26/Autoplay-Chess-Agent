"""Arena gating: Head-to-head match between Candidate and Champion."""

from __future__ import annotations

from typing import Dict, List, Tuple
import chess
import chess.pgn

from rl_chess.mcts.search import MCTS
from rl_chess.net.model import ChessNet


def play_arena_game(
    white_net: ChessNet,
    black_net: ChessNet,
    num_simulations: int = 64,
    temperature: float = 0.2,
    max_plies: int = 200,
    device: str = "auto",
) -> Tuple[str, int]:
    """Play a single arena game between two models.
    
    Returns:
        result: '1-0', '0-1', or '1/2-1/2'
        plies: move count
    """
    mcts_white = MCTS(white_net, device=device)
    mcts_black = MCTS(black_net, device=device)

    board = chess.Board()
    history: List[chess.Board] = []
    ply = 0

    while not board.is_game_over() and ply < max_plies:
        active_mcts = mcts_white if board.turn == chess.WHITE else mcts_black
        move, _, _ = active_mcts.search(
            board=board,
            history=history,
            num_simulations=num_simulations,
            temperature=temperature,
            add_noise=False,
        )

        if move == chess.Move.null() or move not in board.legal_moves:
            break

        history.append(board.copy())
        if len(history) > 8:
            history.pop(0)

        mcts_white.advance_root(move)
        mcts_black.advance_root(move)
        board.push(move)
        ply += 1

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return "1/2-1/2", ply
    if outcome.winner == chess.WHITE:
        return "1-0", ply
    return "0-1", ply


def evaluate_candidate(
    candidate_net: ChessNet,
    champion_net: ChessNet,
    num_games: int = 10,
    num_simulations: int = 64,
    threshold: float = 0.55,
    device: str = "auto",
) -> Tuple[bool, float, List[Dict]]:
    """Play arena match and determine whether to promote candidate.
    
    Returns:
        promoted: bool
        score: float in [0, 1]
        results: list of game records
    """
    candidate_points = 0.0
    game_records = []

    for game_idx in range(num_games):
        # Alternate colors
        if game_idx % 2 == 0:
            white_net, black_net = candidate_net, champion_net
            cand_is_white = True
        else:
            white_net, black_net = champion_net, candidate_net
            cand_is_white = False

        res, plies = play_arena_game(
            white_net=white_net,
            black_net=black_net,
            num_simulations=num_simulations,
            device=device,
        )

        if res == "1/2-1/2":
            pts = 0.5
        elif (res == "1-0" and cand_is_white) or (res == "0-1" and not cand_is_white):
            pts = 1.0
        else:
            pts = 0.0

        candidate_points += pts
        game_records.append({
            "game_idx": game_idx + 1,
            "candidate_color": "white" if cand_is_white else "black",
            "result": res,
            "candidate_points": pts,
            "plies": plies,
        })

    win_rate = candidate_points / max(1, num_games)
    promoted = win_rate >= threshold
    return promoted, win_rate, game_records
