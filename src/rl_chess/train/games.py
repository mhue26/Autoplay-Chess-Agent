"""Self-play data generation and arena gating matches."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import chess
import chess.pgn
import numpy as np

from rl_chess.encoding import encode_board
from rl_chess.game import play_game, result_string, termination_reason
from rl_chess.mcts import MCTS
from rl_chess.model import ChessNet

Sample = Tuple[np.ndarray, np.ndarray, float]  # (encoded state, MCTS pi, outcome z for side to move)


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
    on_move: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_game_end: Optional[Callable[[Dict[str, Any]], None]] = None,
    stop_event: Optional[threading.Event] = None,
) -> Tuple[List[Sample], str, int]:
    """Play one self-play game; returns (training samples, PGN result, plies)."""
    mcts.reset()
    trajectory: List[Tuple[np.ndarray, np.ndarray, chess.Color]] = []

    def pick_move(board: chess.Board, history: List[chess.Board]) -> Optional[chess.Move]:
        if stop_event and stop_event.is_set():
            return None
        ply = len(trajectory)
        explore = ply < temp_decay_plies
        temperature = temp_init if explore else temp_final
        state = encode_board(board, history)
        move, pi, analysis = mcts.search(
            board,
            history,
            num_simulations,
            temperature=temperature,
            add_noise=explore,
            stop_event=stop_event,
        )
        if stop_event and stop_event.is_set():
            return None
        if move == chess.Move.null() or move not in board.legal_moves:
            return None

        trajectory.append((state, pi, board.turn))

        if on_move:
            san = board.san(move)
            b_after = board.copy()
            b_after.push(move)
            root_val = float(analysis.root_value)
            root_val_white = root_val if board.turn == chess.WHITE else -root_val
            legal_pi = pi[pi > 0]
            entropy = -float(np.sum(legal_pi * np.log(legal_pi + 1e-12))) if len(legal_pi) > 0 else 0.0

            candidates_data = [
                {
                    "san": c.san,
                    "uci": c.move.uci(),
                    "from_sq": c.move.from_square,
                    "to_sq": c.move.to_square,
                    "prior": float(c.prior),
                    "visits": int(c.visits),
                    "visit_share": float(c.visit_share),
                    "q_value": float(c.q_value),
                }
                for c in analysis.candidates[:5]
            ]
            on_move({
                "phase": "selfplay",
                "ply": ply + 1,
                "san": san,
                "uci": move.uci(),
                "from_sq": move.from_square,
                "to_sq": move.to_square,
                "mover": "white" if board.turn == chess.WHITE else "black",
                "fen": b_after.fen(),
                "temperature": float(temperature),
                "add_noise": bool(explore),
                "sims": int(num_simulations),
                "root_value": root_val,
                "root_value_white": root_val_white,
                "candidates": candidates_data,
                "pv": [m.uci() for m in getattr(analysis, "principal_variation", [])],
                "pv_san": getattr(analysis, "pv_san", ""),
                "entropy": entropy,
            })

        return move

    board, plies = play_game(
        pick_move,
        max_plies,
        on_move=lambda _board, move: mcts.advance_root(move),
        stop_event=stop_event,
    )
    if stop_event and stop_event.is_set():
        return [], "", plies

    trajectory = trajectory[:plies]  # drop the pending entry if the final search returned no legal move

    result = result_string(board)
    term_reason = termination_reason(board, plies, max_plies)
    white_value = {"1-0": 1.0, "0-1": -1.0}.get(result, 0.0)

    if pgn_path:
        game = chess.pgn.Game.from_board(board)
        game.headers.update({"Event": "Self-Play RL Training", "White": white_name, "Black": black_name, "Result": result})
        path = Path(pgn_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{game}\n\n", encoding="utf-8")

    samples = [(s, pi, white_value if turn == chess.WHITE else -white_value) for s, pi, turn in trajectory]

    if on_game_end:
        on_game_end({
            "phase": "selfplay",
            "result": result,
            "plies": plies,
            "termination_reason": term_reason,
            "white_val": white_value,
            "samples_count": len(samples),
            "final_fen": board.fen(),
        })

    return samples, result, plies


def play_arena_game(
    white_net: ChessNet,
    black_net: ChessNet,
    num_simulations: int = 64,
    temperature: float = 0.2,
    max_plies: int = 200,
    device: str = "auto",
    on_move: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_game_end: Optional[Callable[[Dict[str, Any]], None]] = None,
    stop_event: Optional[threading.Event] = None,
) -> Tuple[str, int]:
    """Play one arena game between two networks; returns (PGN result, plies)."""
    players = {chess.WHITE: MCTS(white_net, device=device), chess.BLACK: MCTS(black_net, device=device)}
    current_ply = 0

    def pick_move(board: chess.Board, history: List[chess.Board]) -> Optional[chess.Move]:
        nonlocal current_ply
        if stop_event and stop_event.is_set():
            return None
        move, pi, analysis = players[board.turn].search(
            board, history, num_simulations, temperature, stop_event=stop_event
        )
        if stop_event and stop_event.is_set():
            return None
        if move == chess.Move.null() or move not in board.legal_moves:
            return None

        if on_move:
            san = board.san(move)
            b_after = board.copy()
            b_after.push(move)
            root_val = float(analysis.root_value)
            root_val_white = root_val if board.turn == chess.WHITE else -root_val
            legal_pi = pi[pi > 0]
            entropy = -float(np.sum(legal_pi * np.log(legal_pi + 1e-12))) if len(legal_pi) > 0 else 0.0

            candidates_data = [
                {
                    "san": c.san,
                    "uci": c.move.uci(),
                    "from_sq": c.move.from_square,
                    "to_sq": c.move.to_square,
                    "prior": float(c.prior),
                    "visits": int(c.visits),
                    "visit_share": float(c.visit_share),
                    "q_value": float(c.q_value),
                }
                for c in analysis.candidates[:5]
            ]
            on_move({
                "phase": "arena",
                "ply": current_ply + 1,
                "san": san,
                "uci": move.uci(),
                "from_sq": move.from_square,
                "to_sq": move.to_square,
                "mover": "white" if board.turn == chess.WHITE else "black",
                "fen": b_after.fen(),
                "temperature": float(temperature),
                "add_noise": False,
                "sims": int(num_simulations),
                "root_value": root_val,
                "root_value_white": root_val_white,
                "candidates": candidates_data,
                "pv": [m.uci() for m in getattr(analysis, "principal_variation", [])],
                "pv_san": getattr(analysis, "pv_san", ""),
                "entropy": entropy,
            })

        current_ply += 1
        return move

    def advance(_board: chess.Board, move: chess.Move) -> None:
        for p in players.values():
            p.advance_root(move)

    board, plies = play_game(pick_move, max_plies, on_move=advance, stop_event=stop_event)
    if stop_event and stop_event.is_set():
        return "", plies
    res = result_string(board)
    term_reason = termination_reason(board, plies, max_plies)

    if on_game_end:
        on_game_end({
            "phase": "arena",
            "result": res,
            "plies": plies,
            "termination_reason": term_reason,
            "final_fen": board.fen(),
        })

    return res, plies


def evaluate_candidate(
    candidate_net: ChessNet,
    champion_net: ChessNet,
    num_games: int = 10,
    num_simulations: int = 64,
    threshold: float = 0.55,
    device: str = "auto",
    on_game_start: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_move: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_game_end: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_game_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
    cand_name: str = "Candidate",
    champ_name: str = "Champion",
    stop_event: Optional[threading.Event] = None,
) -> Tuple[bool, float, List[Dict]]:
    """Arena match with alternating colors; returns (promoted, candidate score in [0, 1], game records)."""
    points = 0.0
    records = []

    for i in range(num_games):
        if stop_event and stop_event.is_set():
            break
        cand_is_white = i % 2 == 0
        white_net, black_net = (candidate_net, champion_net) if cand_is_white else (champion_net, candidate_net)
        white_name = cand_name if cand_is_white else champ_name
        black_name = champ_name if cand_is_white else cand_name

        game_idx = i + 1
        candidate_color = "white" if cand_is_white else "black"
        meta = {
            "phase": "arena",
            "game_idx": game_idx,
            "total_games": num_games,
            "candidate_color": candidate_color,
        }

        if on_game_start:
            on_game_start({
                **meta,
                "white": white_name,
                "black": black_name,
            })

        def move_cb(m_data: Dict[str, Any]) -> None:
            if on_move:
                on_move({**m_data, **meta})

        def game_end_cb(ge_data: Dict[str, Any]) -> None:
            if on_game_end:
                on_game_end({**ge_data, **meta})

        res, plies = play_arena_game(
            white_net,
            black_net,
            num_simulations,
            device=device,
            on_move=move_cb,
            on_game_end=game_end_cb,
            stop_event=stop_event,
        )

        if stop_event and stop_event.is_set():
            break

        pts = 0.5 if res == "1/2-1/2" else float((res == "1-0") == cand_is_white)
        points += pts
        rec = {
            "game_idx": game_idx,
            "total_games": num_games,
            "candidate_color": candidate_color,
            "result": res,
            "candidate_points": pts,
            "plies": plies,
            "running_score": points / (i + 1),
        }
        records.append(rec)
        if on_game_complete:
            on_game_complete(rec)

    if stop_event and stop_event.is_set():
        return False, (points / max(1, len(records))) if records else 0.0, records
    win_rate = points / max(1, num_games)
    return win_rate >= threshold, win_rate, records
