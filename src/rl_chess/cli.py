"""Command-line interface for RL-Chess."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rl_chess.ledger.database import Ledger


def cmd_stats(args: argparse.Namespace) -> None:
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Ledger database not found at {db_path}. No training history yet.")
        return

    ledger = Ledger(db_path)
    summary = ledger.get_stats_summary()

    print("=" * 60)
    print(" RL-CHESS TRAINING & EVALUATION LEDGER")
    print("=" * 60)
    print(f"Total Games Played:        {summary['total_games']}")
    print(f"  • Self-Play Games:       {summary['selfplay_games']} (~{summary['selfplay_positions_estimated']} positions)")
    print(f"  • Arena Gating Games:    {summary['arena_games']}")
    print(f"  • Benchmark Games:       {summary['benchmark_games']}")
    print(f"  • Human Matches:         {summary['human_games']}")
    print(f"Total Training Steps:      {summary['training_steps']}")
    print(f"Registered Checkpoints:    {summary['checkpoints_count']}")

    champ = summary["champion"]
    if champ:
        print(f"Current Champion:          {champ['name']} (Gen {champ['generation']})")
        print(f"  Path:                    {champ['path']}")
    else:
        print("Current Champion:          None (untrained)")

    elo_hist = ledger.get_elo_history()
    if elo_hist:
        print("\nElo Rating History:")
        print(f"{'Checkpoint':<20} {'Gen':<5} {'Method':<18} {'Elo':<10} {'CI (95%)':<16} {'Games':<6}")
        print("-" * 80)
        for row in elo_hist:
            ci = f"[{row['ci_low']:.0f}, {row['ci_high']:.0f}]" if row['ci_low'] is not None and row['ci_high'] is not None else "-"
            print(f"{row['checkpoint_name']:<20} {row['generation']:<5} {row['method']:<18} {row['elo']:<10.1f} {ci:<16} {row['n_games']:<6}")
    else:
        print("\nElo Rating History:        No benchmarks recorded yet.")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(prog="rl-chess", description="RL Chess Agent CLI")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # stats
    stats_p = subparsers.add_parser("stats", help="Show training and Elo ledger summary")
    stats_p.add_argument("--db", default="data/ledger.db", help="Path to SQLite ledger file")

    # train
    train_p = subparsers.add_parser("train", help="Run self-play training loop")
    train_p.add_argument("--config", default="configs/tiny.yaml", help="Path to YAML config")
    train_p.add_argument("--generations", type=int, default=1, help="Number of generations to run")

    # benchmark
    bench_p = subparsers.add_parser("benchmark", help="Run Stockfish benchmark and estimate Elo")
    bench_p.add_argument("--checkpoint", type=str, default="", help="Checkpoint path or name (defaults to champion)")
    bench_p.add_argument("--config", default="configs/tiny.yaml", help="Path to YAML config")
    bench_p.add_argument("--games", type=int, default=10, help="Games per opponent rung")

    # play
    play_p = subparsers.add_parser("play", help="Launch pygame UI to play against the agent")
    play_p.add_argument("--checkpoint", type=str, default="", help="Checkpoint path (defaults to champion/random)")
    play_p.add_argument("--config", default="configs/tiny.yaml", help="Path to YAML config")
    play_p.add_argument("--color", choices=["white", "black"], default="white", help="Human color")

    args = parser.parse_args()
    if args.command == "stats":
        cmd_stats(args)
    elif args.command == "train":
        from rl_chess.train.loop import run_training
        run_training(args.config, args.generations)
    elif args.command == "benchmark":
        from rl_chess.eval.benchmark import run_benchmark
        run_benchmark(args.config, args.checkpoint, args.games)
    elif args.command == "play":
        from rl_chess.gui.app import run_app
        run_app(args.config, args.checkpoint, args.color)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
