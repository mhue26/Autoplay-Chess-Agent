"""Stockfish benchmark gauntlet and human-meaningful Elo evaluation."""

from __future__ import annotations

import os
import random
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import chess
import chess.engine
import torch
import yaml

from rl_chess.eval.elo import MatchResult, fit_elo_bradley_terry
from rl_chess.ledger.database import Ledger
from rl_chess.mcts.search import MCTS
from rl_chess.net.model import create_model, resolve_device


class OpponentRung:
    def __init__(self, name: str, elo: float, skill_level: Optional[int] = None, uci_elo: Optional[int] = None):
        self.name = name
        self.elo = elo
        self.skill_level = skill_level
        self.uci_elo = uci_elo


# Calibrated ladder rungs
DEFAULT_LADDER = [
    OpponentRung("Random-Player", elo=200.0),
    OpponentRung("Stockfish-Skill-0", elo=800.0, skill_level=0),
    OpponentRung("Stockfish-Skill-3", elo=1100.0, skill_level=3),
    OpponentRung("Stockfish-Skill-6", elo=1400.0, skill_level=6),
    OpponentRung("Stockfish-Elo-1320", elo=1320.0, uci_elo=1320),
    OpponentRung("Stockfish-Elo-1500", elo=1500.0, uci_elo=1500),
]


def find_stockfish_executable(configured_path: str = "") -> Optional[str]:
    candidates = [
        configured_path,
        "/opt/homebrew/bin/stockfish",
        "/usr/local/bin/stockfish",
        "stockfish",
    ]
    for c in candidates:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return c
    return None


def play_benchmark_game(
    mcts: MCTS,
    engine: Optional[chess.engine.SimpleEngine],
    rung: OpponentRung,
    agent_is_white: bool,
    num_simulations: int = 64,
    time_limit_sec: float = 0.1,
    max_plies: int = 200,
) -> Tuple[str, int]:
    """Play one benchmark game between agent and an opponent rung."""
    board = chess.Board()
    history: List[chess.Board] = []
    ply = 0
    mcts.reset()

    while not board.is_game_over() and ply < max_plies:
        turn_is_agent = (board.turn == chess.WHITE and agent_is_white) or (board.turn == chess.BLACK and not agent_is_white)

        if turn_is_agent:
            move, _, _ = mcts.search(
                board=board,
                history=history,
                num_simulations=num_simulations,
                temperature=0.0,  # Competitive
                add_noise=False,
            )
        else:
            if engine is None or rung.name == "Random-Player":
                move = random.choice(list(board.legal_moves))
            else:
                result = engine.play(board, chess.engine.Limit(time=time_limit_sec))
                move = result.move if result.move else random.choice(list(board.legal_moves))

        if move == chess.Move.null() or move not in board.legal_moves:
            break

        history.append(board.copy())
        if len(history) > 8:
            history.pop(0)

        mcts.advance_root(move)
        board.push(move)
        ply += 1

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return "1/2-1/2", ply
    if outcome.winner == chess.WHITE:
        return "1-0", ply
    return "0-1", ply


def run_benchmark(
    config_path: str = "configs/tiny.yaml",
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

    cfg_file = Path(config_path)
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    bench_cfg = cfg.get("benchmark", {})
    paths_cfg = cfg.get("paths", {})
    ledger = Ledger(paths_cfg.get("ledger_db", "data/ledger.db"))
    device = resolve_device(cfg.get("device", "auto"))

    # Resolve checkpoint
    if checkpoint_target:
        cp_path = Path(checkpoint_target)
        if not cp_path.exists():
            cp_rec = next((c for c in ledger.list_checkpoints() if c.name == checkpoint_target), None)
            if cp_rec:
                cp_path = Path(cp_rec.path)
                cp_id = cp_rec.id
            else:
                raise FileNotFoundError(f"Checkpoint {checkpoint_target} not found.")
        else:
            cp_rec = next((c for c in ledger.list_checkpoints() if c.path == str(cp_path)), None)
            cp_id = cp_rec.id if cp_rec else 1
        cp_name = cp_path.stem
    else:
        champ = ledger.get_champion_checkpoint()
        if champ is None:
            raise RuntimeError("No checkpoint registered in ledger. Run training first.")
        cp_path = Path(champ.path)
        cp_name = champ.name
        cp_id = champ.id

    log(f"Benchmarking checkpoint: {cp_name} ({cp_path})")
    net = create_model(profile=cfg.get("profile", "tiny"), device=str(device))
    net.load_state_dict(torch.load(cp_path, map_location=device))
    net.eval()

    num_sims = bench_cfg.get("agent_simulations", 64)
    time_limit = bench_cfg.get("time_limit_sec", 0.1)
    mcts = MCTS(net, device=str(device))

    # Setup Stockfish
    sf_path = find_stockfish_executable(bench_cfg.get("stockfish_path", ""))
    if sf_path:
        log(f"Stockfish engine found at: {sf_path}")
    else:
        log("Warning: Stockfish engine binary not found. Benchmark will use Random-Player anchor only.")

    # Select rungs
    rungs_to_run = [DEFAULT_LADDER[0]]  # Random anchor
    if sf_path:
        # Include skill 0 and skill 3 for fast baseline
        rungs_to_run.extend([DEFAULT_LADDER[1], DEFAULT_LADDER[2]])

    total_expected = len(rungs_to_run) * games_per_opponent
    games_completed = 0
    match_results: List[MatchResult] = []

    for rung in rungs_to_run:
        if stop_event and stop_event.is_set():
            log("Benchmark stopped by user request.")
            break

        log(f"\n--- Running gauntlet vs {rung.name} (Anchor Elo: {rung.elo:.0f}) ---")
        engine = None
        if sf_path and rung.name != "Random-Player":
            engine = chess.engine.SimpleEngine.popen_uci(sf_path)
            if rung.skill_level is not None:
                engine.configure({"Skill Level": rung.skill_level})
            elif rung.uci_elo is not None:
                engine.configure({"UCI_LimitStrength": True, "UCI_Elo": rung.uci_elo})

        wins = 0
        draws = 0
        losses = 0

        try:
            for g in range(games_per_opponent):
                if stop_event and stop_event.is_set():
                    log("Benchmark stopped during match.")
                    break

                agent_white = (g % 2 == 0)
                res, plies = play_benchmark_game(
                    mcts=mcts,
                    engine=engine,
                    rung=rung,
                    agent_is_white=agent_white,
                    num_simulations=num_sims,
                    time_limit_sec=time_limit,
                )

                if res == "1/2-1/2":
                    draws += 1
                elif (res == "1-0" and agent_white) or (res == "0-1" and not agent_white):
                    wins += 1
                else:
                    losses += 1

                ledger.record_game(
                    kind="benchmark",
                    white=cp_name if agent_white else rung.name,
                    black=rung.name if agent_white else cp_name,
                    result=res,
                    plies=plies,
                    checkpoint_id=cp_id,
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

        match = MatchResult(
            opponent_name=rung.name,
            opponent_elo=rung.elo,
            wins=wins,
            draws=draws,
            losses=losses,
        )
        match_results.append(match)
        log(f"Result vs {rung.name}: {wins}W - {draws}D - {losses}L (Score: {match.score}/{match.total_games})")

    if not match_results or sum(m.total_games for m in match_results) == 0:
        return None

    # Fit Bradley-Terry Elo
    estimate = fit_elo_bradley_terry(match_results)
    log("\n" + "=" * 55)
    log(" BENCHMARK ELO ESTIMATE")
    log("=" * 55)
    log(f"Fitted Elo:            {estimate.rating:.1f}")
    log(f"95% Confidence Interval: [{estimate.ci_low:.1f}, {estimate.ci_high:.1f}] (±{1.96 * estimate.std_err:.1f})")
    log(f"Total Games Played:    {estimate.total_games}")
    log(f"Total Score:           {estimate.total_score:.1f}/{estimate.total_games}")
    log("=" * 55)

    # Record to ledger
    opponents_summary = ",".join(m.opponent_name for m in match_results)
    ledger.record_elo_estimate(
        checkpoint_id=cp_id,
        method="stockfish_anchor",
        elo=estimate.rating,
        n_games=estimate.total_games,
        ci_low=estimate.ci_low,
        ci_high=estimate.ci_high,
        opponent_set=opponents_summary,
        budget_notes=f"sims={num_sims}, time_limit={time_limit}s",
    )
    log("Saved Elo estimate to ledger database.")

    result_dict = {
        "checkpoint_id": cp_id,
        "checkpoint_name": cp_name,
        "elo": estimate.rating,
        "ci_low": estimate.ci_low,
        "ci_high": estimate.ci_high,
        "std_err": estimate.std_err,
        "total_games": estimate.total_games,
        "total_score": estimate.total_score,
        "opponents": [
            {
                "name": m.opponent_name,
                "elo": m.opponent_elo,
                "wins": m.wins,
                "draws": m.draws,
                "losses": m.losses,
                "score": m.score,
            }
            for m in match_results
        ],
    }
    if on_result:
        on_result(result_dict)
    return result_dict
