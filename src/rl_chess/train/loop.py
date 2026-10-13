"""AlphaZero self-play training loop orchestrator."""

from __future__ import annotations

import copy
import hashlib
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional
import torch
import yaml

from rl_chess.ledger.database import Ledger
from rl_chess.mcts.search import MCTS
from rl_chess.net.model import create_model, resolve_device
from rl_chess.train.arena import evaluate_candidate
from rl_chess.train.replay import ReplayBuffer
from rl_chess.train.selfplay import play_selfplay_game
from rl_chess.train.trainer import Trainer


def compute_file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_training(
    config_path: str = "configs/tiny.yaml",
    generations: int = 1,
    on_log: Optional[Callable[[str], None]] = None,
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    stop_event: Optional[threading.Event] = None,
) -> None:
    def log(msg: str) -> None:
        print(msg)
        if on_log:
            on_log(msg)

    def report_progress(data: Dict[str, Any]) -> None:
        if on_progress:
            on_progress(data)

    cfg_file = Path(config_path)
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    paths_cfg = cfg.get("paths", {})
    ledger_db = Path(paths_cfg.get("ledger_db", "data/ledger.db"))
    checkpoints_dir = Path(paths_cfg.get("checkpoints_dir", "data/checkpoints"))
    replay_dir = Path(paths_cfg.get("replay_dir", "data/replay"))
    pgns_dir = Path(paths_cfg.get("pgns_dir", "data/pgns"))

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    replay_dir.mkdir(parents=True, exist_ok=True)
    pgns_dir.mkdir(parents=True, exist_ok=True)

    ledger = Ledger(ledger_db)
    device_str = cfg.get("device", "auto")
    device = resolve_device(device_str)
    profile = cfg.get("profile", "tiny")

    # 1. Initialize or load champion model
    champion_rec = ledger.get_champion_checkpoint()
    champion_net = create_model(profile=profile, device=str(device))

    if champion_rec is not None and Path(champion_rec.path).exists():
        log(f"Loading champion from {champion_rec.path} (Gen {champion_rec.generation})")
        state = torch.load(champion_rec.path, map_location=device)
        champion_net.load_state_dict(state)
        current_gen = champion_rec.generation
        champion_id = champion_rec.id
    else:
        log("No existing champion found. Initializing Generation 0...")
        current_gen = 0
        gen0_path = checkpoints_dir / "gen_0.pt"
        torch.save(champion_net.state_dict(), gen0_path)
        sha = compute_file_sha256(gen0_path)
        champion_id = ledger.record_checkpoint(
            name="gen_0",
            generation=0,
            path=str(gen0_path),
            sha256=sha,
            promoted=True,
            config=cfg,
        )
        ledger.promote_checkpoint(champion_id)

    # 2. Replay buffer
    train_cfg = cfg.get("training", {})
    replay_buffer = ReplayBuffer(capacity=train_cfg.get("replay_buffer_capacity", 50000))
    buffer_file = replay_dir / "replay_buffer.npz"
    if buffer_file.exists():
        log(f"Loading existing replay buffer from {buffer_file}...")
        replay_buffer.load(buffer_file)
        log(f"Replay buffer loaded with {len(replay_buffer)} positions.")

    # 3. Main training generations
    selfplay_cfg = cfg.get("selfplay", {})
    mcts_cfg = cfg.get("mcts", {})
    arena_cfg = cfg.get("arena", {})

    games_per_gen = selfplay_cfg.get("games_per_generation", 10)
    sims_selfplay = mcts_cfg.get("num_simulations", 64)
    train_steps_per_gen = train_cfg.get("train_steps_per_generation", 100)
    batch_size = train_cfg.get("batch_size", 64)

    for g_idx, g in enumerate(range(current_gen + 1, current_gen + 1 + generations), start=1):
        if stop_event and stop_event.is_set():
            log("Training stopped by user request.")
            break

        log(f"\n{'=' * 50}")
        log(f" STARTING GENERATION {g} ({g_idx}/{generations})")
        log(f"{'=' * 50}")

        # A. Self-Play
        log(f"[Self-Play] Generating {games_per_gen} games with champion (Gen {current_gen})...")
        mcts_champ = MCTS(
            champion_net,
            c_puct=mcts_cfg.get("c_puct", 1.5),
            dirichlet_alpha=mcts_cfg.get("dirichlet_alpha", 0.3),
            dirichlet_eps=mcts_cfg.get("dirichlet_epsilon", 0.25),
            device=str(device),
        )

        for game_idx in range(1, games_per_gen + 1):
            if stop_event and stop_event.is_set():
                log("Training stopped during self-play.")
                break

            pgn_file = pgns_dir / f"selfplay_gen{g}_game{game_idx}.pgn"
            samples, result, plies = play_selfplay_game(
                mcts=mcts_champ,
                num_simulations=sims_selfplay,
                temp_init=mcts_cfg.get("temp_init", 1.0),
                temp_final=mcts_cfg.get("temp_final", 0.0),
                temp_decay_plies=mcts_cfg.get("temp_decay_plies", 30),
                max_plies=selfplay_cfg.get("max_plies", 200),
                pgn_path=pgn_file,
                white_name=f"Champion-Gen{current_gen}",
                black_name=f"Champion-Gen{current_gen}",
            )

            replay_buffer.push_game(samples)
            ledger.record_game(
                kind="selfplay",
                white=f"Champion-Gen{current_gen}",
                black=f"Champion-Gen{current_gen}",
                result=result,
                plies=plies,
                pgn_path=str(pgn_file),
                checkpoint_id=champion_id,
            )
            log(f"  Game {game_idx}/{games_per_gen}: {result} ({plies} plies, {len(samples)} samples)")
            report_progress({
                "phase": "Self-Play",
                "generation": g,
                "current": game_idx,
                "total": games_per_gen,
                "fraction": game_idx / max(1, games_per_gen),
            })

        if stop_event and stop_event.is_set():
            break

        log(f"[Replay] Buffer now holds {len(replay_buffer)} positions.")
        replay_buffer.save(buffer_file)

        # B. Training Candidate Model
        log(f"[Training] Training candidate network for {train_steps_per_gen} steps...")
        candidate_net = copy.deepcopy(champion_net)
        trainer = Trainer(
            candidate_net,
            learning_rate=train_cfg.get("learning_rate", 0.001),
            weight_decay=train_cfg.get("weight_decay", 0.0001),
            device=str(device),
        )

        for step in range(1, train_steps_per_gen + 1):
            if stop_event and stop_event.is_set():
                log("Training stopped during optimization.")
                break

            metrics = trainer.train_step(replay_buffer, batch_size=batch_size)
            if step % max(1, train_steps_per_gen // 5) == 0 or step == train_steps_per_gen:
                log(f"  Step {step}/{train_steps_per_gen} - loss: {metrics['loss_total']:.4f} (p: {metrics['loss_policy']:.4f}, v: {metrics['loss_value']:.4f})")
                ledger.record_train_step(
                    generation=g,
                    step=step,
                    loss_total=metrics["loss_total"],
                    loss_policy=metrics["loss_policy"],
                    loss_value=metrics["loss_value"],
                    lr=metrics["lr"],
                    checkpoint_id=champion_id,
                )
            report_progress({
                "phase": "Optimization",
                "generation": g,
                "current": step,
                "total": train_steps_per_gen,
                "fraction": step / max(1, train_steps_per_gen),
            })

        if stop_event and stop_event.is_set():
            break

        # Save candidate checkpoint
        cand_path = checkpoints_dir / f"gen_{g}.pt"
        torch.save(candidate_net.state_dict(), cand_path)
        sha = compute_file_sha256(cand_path)
        cand_id = ledger.record_checkpoint(
            name=f"gen_{g}",
            generation=g,
            parent_id=champion_id,
            path=str(cand_path),
            sha256=sha,
            promoted=False,
            config=cfg,
        )

        # C. Arena Gating
        arena_games = arena_cfg.get("num_games", 10)
        threshold = arena_cfg.get("win_rate_threshold", 0.55)
        log(f"[Arena] Gating match: Candidate Gen {g} vs Champion Gen {current_gen} ({arena_games} games)...")

        promoted, win_rate, records = evaluate_candidate(
            candidate_net=candidate_net,
            champion_net=champion_net,
            num_games=arena_games,
            num_simulations=sims_selfplay,
            threshold=threshold,
            device=str(device),
        )

        for rec in records:
            ledger.record_game(
                kind="arena",
                white=f"gen_{g}" if rec["candidate_color"] == "white" else f"gen_{current_gen}",
                black=f"gen_{current_gen}" if rec["candidate_color"] == "white" else f"gen_{g}",
                result=rec["result"],
                plies=rec["plies"],
                checkpoint_id=cand_id,
            )

        log(f"[Arena Result] Candidate score: {win_rate * 100:.1f}% (threshold: {threshold * 100:.1f}%)")
        if promoted:
            log(f"🎉 Candidate Gen {g} PROMOTED to Champion!")
            ledger.promote_checkpoint(cand_id)
            champion_net = candidate_net
            champion_id = cand_id
            current_gen = g
            report_progress({
                "phase": "Promoted",
                "generation": g,
                "champion_id": cand_id,
                "champion_name": f"gen_{g}",
            })
        else:
            log(f"Candidate Gen {g} rejected. Champion remains Gen {current_gen}.")

    log("\nTraining run finished.")
