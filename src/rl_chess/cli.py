"""Command-line interface: `rl-chess` launches the dashboard; train / benchmark / stats run headless."""

from __future__ import annotations

import argparse
from pathlib import Path

from rl_chess.config import DEFAULT_CONFIG_PATH
from rl_chess.ledger import Ledger


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
            has_ci = row["ci_low"] is not None and row["ci_high"] is not None
            ci = f"[{row['ci_low']:.0f}, {row['ci_high']:.0f}]" if has_ci else "-"
            print(f"{row['checkpoint_name']:<20} {row['generation']:<5} {row['method']:<18} {row['elo']:<10.1f} {ci:<16} {row['n_games']:<6}")
    else:
        print("\nElo Rating History:        No benchmarks recorded yet.")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rl-chess",
        description="RL Chess Agent: AlphaZero engine and desktop dashboard (launched when no sub-command is given).",
    )

    def add_config(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to YAML config")

    add_config(parser)
    parser.add_argument("--checkpoint", default="", help="Checkpoint name (defaults to champion)")
    parser.add_argument("--color", choices=["white", "black"], default="white", help="Human color in the Play tab")

    sub = parser.add_subparsers(dest="command")

    stats_p = sub.add_parser("stats", help="Show training and Elo ledger summary")
    stats_p.add_argument("--db", default="data/ledger.db", help="Path to SQLite ledger file")

    train_p = sub.add_parser("train", help="Run self-play training loop (headless)")
    add_config(train_p)
    train_p.add_argument("--generations", type=int, default=1, help="Number of generations to run")
    train_p.add_argument(
        "--games-per-gen",
        type=int,
        default=None,
        help="Self-play games before each training weight update (default: config.yaml)",
    )
    train_p.add_argument(
        "--reset-agent",
        action="store_true",
        help="Clear replay buffer, ledger, and all checkpoint files before training",
    )

    bench_p = sub.add_parser("benchmark", help="Run Stockfish benchmark and estimate Elo (headless)")
    add_config(bench_p)
    bench_p.add_argument("--checkpoint", default="", help="Checkpoint name (defaults to champion)")
    bench_p.add_argument("--games", type=int, default=10, help="Games per opponent rung")

    args = parser.parse_args()

    if args.command == "stats":
        cmd_stats(args)
    elif args.command == "train":
        from rl_chess.config import get_paths, load_config
        from rl_chess.ledger import Ledger
        from rl_chess.train.loop import run_training
        from rl_chess.train.reset import reset_agent_to_gen_zero

        if args.reset_agent:
            cfg = load_config(args.config)
            paths = get_paths(cfg)
            reset_agent_to_gen_zero(cfg, paths, Ledger(paths["ledger_db"]))
        run_training(args.config, args.generations, games_per_generation=args.games_per_gen)
    elif args.command == "benchmark":
        from rl_chess.benchmark import run_benchmark
        run_benchmark(args.config, args.checkpoint, args.games)
    else:
        from rl_chess.gui.app import run_app
        run_app(args.config, args.checkpoint, args.color)


if __name__ == "__main__":
    main()
