"""Stockfish benchmark gauntlet: play fixed opponents and fit a Bradley-Terry Elo."""

from __future__ import annotations

import os
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import chess
import chess.engine

from rl_chess.config import DEFAULT_CONFIG_PATH, get_paths, load_config
from rl_chess.elo import MatchResult, fit_elo_bradley_terry
from rl_chess.game import play_game, result_string
from rl_chess.ledger import Ledger
from rl_chess.mcts import MCTS
from rl_chess.model import load_weights, model_from_config


@dataclass
class OpponentRung:
    name: str
    elo: float
    skill_level: Optional[int] = None


# Ladder rungs with nominal Elo anchors.
LADDER = [
    OpponentRung("Random-Player", elo=200.0),
    OpponentRung("Stockfish-Skill-0", elo=800.0, skill_level=0),
    OpponentRung("Stockfish-Skill-3", elo=1100.0, skill_level=3),
]


def find_stockfish_executable(configured_path: str = "") -> Optional[str]:
    candidates = [configured_path, "/opt/homebrew/bin/stockfish", "/usr/local/bin/stockfish", "stockfish"]
    return next((c for c in candidates if c and Path(c).is_file() and os.access(c, os.X_OK)), None)


def play_benchmark_game(
    mcts: MCTS,
    engine: Optional[chess.engine.SimpleEngine],
    rung: OpponentRung,
    agent_is_white: bool,
    num_simulations: int = 64,
    time_limit_sec: float = 0.1,
    max_plies: int = 200,
) -> Tuple[str, int]:
    """Play one game between the agent and an opponent rung; returns (PGN result, plies)."""
    mcts.reset()

    def pick_move(board: chess.Board, history: List[chess.Board]) -> Optional[chess.Move]:
        if (board.turn == chess.WHITE) == agent_is_white:
            move, _, _ = mcts.search(board, history, num_simulations, temperature=0.0)
            return move
        if engine is None or rung.name == "Random-Player":
            return random.choice(list(board.legal_moves))
        played = engine.play(board, chess.engine.Limit(time=time_limit_sec)).move
        return played or random.choice(list(board.legal_moves))

    board, plies = play_game(pick_move, max_plies, on_move=lambda _board, move: mcts.advance_root(move))
    return result_string(board), plies


def run_benchmark(
    config_path: str = DEFAULT_CONFIG_PATH,
    checkpoint_target: str = "",
    games_per_opponent: int = 10,
    on_log: Optional[Callable[[str], None]] = None,
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
    stop_event: Optional[threading.Event] = None,
) -> Optional[Dict[str, Any]]:
    def log(msg: str) -> None:
        print(msg)
        if on_log:
            on_log(msg)

    def stopped() -> bool:
        return bool(stop_event and stop_event.is_set())

    cfg = load_config(config_path)
    bench_cfg = cfg.get("benchmark", {})
    ledger = Ledger(get_paths(cfg)["ledger_db"])

    cp = ledger.resolve_checkpoint(checkpoint_target)
    if cp is None:
        raise RuntimeError(
            f"Checkpoint {checkpoint_target!r} not found." if checkpoint_target
            else "No checkpoint registered in ledger. Run training first."
        )
    log(f"Benchmarking checkpoint: {cp.name} ({cp.path})")
    net = model_from_config(cfg)
    load_weights(net, cp.path)

    num_sims = bench_cfg.get("agent_simulations", 64)
    time_limit = bench_cfg.get("time_limit_sec", 0.1)
    mcts = MCTS(net, device=cfg.get("device", "auto"))

    sf_path = find_stockfish_executable(bench_cfg.get("stockfish_path", ""))
    if sf_path:
        log(f"Stockfish engine found at: {sf_path}")
        rungs = LADDER
    else:
        log("Warning: Stockfish engine binary not found. Benchmark will use Random-Player anchor only.")
        rungs = LADDER[:1]

    total_expected = len(rungs) * games_per_opponent
    games_completed = 0
    match_results: List[MatchResult] = []

    for rung in rungs:
        if stopped():
            log("Benchmark stopped by user request.")
            break

        log(f"\n--- Running gauntlet vs {rung.name} (Anchor Elo: {rung.elo:.0f}) ---")
        engine = None
        if sf_path and rung.skill_level is not None:
            engine = chess.engine.SimpleEngine.popen_uci(sf_path)
            engine.configure({"Skill Level": rung.skill_level})

        wins = draws = losses = 0
        try:
            for g in range(games_per_opponent):
                if stopped():
                    log("Benchmark stopped during match.")
                    break

                agent_white = g % 2 == 0
                res, plies = play_benchmark_game(mcts, engine, rung, agent_white, num_sims, time_limit)

                if res == "1/2-1/2":
                    draws += 1
                elif (res == "1-0") == agent_white:
                    wins += 1
                else:
                    losses += 1

                ledger.record_game(
                    kind="benchmark",
                    white=cp.name if agent_white else rung.name,
                    black=rung.name if agent_white else cp.name,
                    result=res,
                    plies=plies,
                    checkpoint_id=cp.id,
                )
                games_completed += 1
                log(f"  Game {g + 1}/{games_per_opponent}: {res} ({'White' if agent_white else 'Black'})")
                if on_progress:
                    on_progress({
                        "completed": games_completed,
                        "total": total_expected,
                        "fraction": games_completed / max(1, total_expected),
                        "rung": rung.name,
                        "wins": wins,
                        "draws": draws,
                        "losses": losses,
                    })
        finally:
            if engine:
                engine.quit()

        match = MatchResult(rung.name, rung.elo, wins, draws, losses)
        match_results.append(match)
        log(f"Result vs {rung.name}: {wins}W - {draws}D - {losses}L (Score: {match.score}/{match.total_games})")

    if sum(m.total_games for m in match_results) == 0:
        return None

    estimate = fit_elo_bradley_terry(match_results)
    log("\n" + "=" * 55)
    log(" BENCHMARK ELO ESTIMATE")
    log("=" * 55)
    log(f"Fitted Elo:            {estimate.rating:.1f}")
    log(f"95% Confidence Interval: [{estimate.ci_low:.1f}, {estimate.ci_high:.1f}] (±{1.96 * estimate.std_err:.1f})")
    log(f"Total Games Played:    {estimate.total_games}")
    log(f"Total Score:           {estimate.total_score:.1f}/{estimate.total_games}")
    log("=" * 55)

    ledger.record_elo_estimate(
        checkpoint_id=cp.id,
        method="stockfish_anchor",
        elo=estimate.rating,
        n_games=estimate.total_games,
        ci_low=estimate.ci_low,
        ci_high=estimate.ci_high,
        opponent_set=",".join(m.opponent_name for m in match_results),
        budget_notes=f"sims={num_sims}, time_limit={time_limit}s",
    )
    log("Saved Elo estimate to ledger database.")

    result = {
        "checkpoint_id": cp.id,
        "checkpoint_name": cp.name,
        "elo": estimate.rating,
        "ci_low": estimate.ci_low,
        "ci_high": estimate.ci_high,
        "std_err": estimate.std_err,
        "total_games": estimate.total_games,
        "total_score": estimate.total_score,
        "opponents": [
            {"name": m.opponent_name, "elo": m.opponent_elo, "wins": m.wins,
             "draws": m.draws, "losses": m.losses, "score": m.score}
            for m in match_results
        ],
    }
    if on_result:
        on_result(result)
    return result
