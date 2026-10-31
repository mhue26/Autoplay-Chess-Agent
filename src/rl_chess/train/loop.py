"""AlphaZero loop: self-play -> train candidate -> arena gate -> promote."""

from __future__ import annotations

import copy
import hashlib
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch

from rl_chess.config import DEFAULT_CONFIG_PATH, get_paths, load_config
from rl_chess.ledger import Ledger
from rl_chess.mcts import MCTS
from rl_chess.model import load_weights, model_from_config
from rl_chess.train.games import evaluate_candidate, play_selfplay_game
from rl_chess.train.replay import ReplayBuffer
from rl_chess.train.trainer import Trainer


def resolve_arena_games(games_per_gen: int, arena_cfg: Dict[str, Any]) -> int:
    """Arena match length tracks self-play batch size, capped (default 20)."""
    cap = int(arena_cfg.get("max_games", arena_cfg.get("num_games", 20)))
    return min(max(1, int(games_per_gen)), max(1, cap))


def compute_file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_training(
    config_path: str = DEFAULT_CONFIG_PATH,
    generations: int = 1,
    games_per_generation: Optional[int] = None,
    on_log: Optional[Callable[[str], None]] = None,
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    stop_event: Optional[threading.Event] = None,
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    def log(msg: str) -> None:
        print(msg)
        if on_log:
            on_log(msg)

    def emit(event_type: str, **kwargs: Any) -> None:
        if on_event:
            on_event({"type": event_type, **kwargs})

    def progress(phase: str, generation: int, current: int, total: int) -> None:
        if on_progress:
            on_progress({
                "phase": phase,
                "generation": generation,
                "current": current,
                "total": total,
                "fraction": current / max(1, total),
            })

    def stopped(where: str = "") -> bool:
        if stop_event and stop_event.is_set():
            log(f"Training stopped{where}.")
            return True
        return False

    cfg = load_config(config_path)
    paths = get_paths(cfg)
    checkpoints_dir = Path(paths["checkpoints_dir"])
    replay_dir = Path(paths["replay_dir"])
    pgns_dir = Path(paths["pgns_dir"])
    for d in (checkpoints_dir, replay_dir, pgns_dir):
        d.mkdir(parents=True, exist_ok=True)

    ledger = Ledger(paths["ledger_db"])
    device = cfg.get("device", "auto")

    # 1. Load (or initialise) the champion.
    champion_rec = ledger.get_champion_checkpoint()
    champion_net = model_from_config(cfg)

    if champion_rec is not None and Path(champion_rec.path).exists():
        log(f"Loading champion from {champion_rec.path} (Gen {champion_rec.generation})")
        load_weights(champion_net, champion_rec.path)
        current_gen = champion_rec.generation
        champion_id = champion_rec.id
    else:
        log("No existing champion found. Initializing Generation 0...")
        current_gen = 0
        gen0_path = checkpoints_dir / "gen_0.pt"
        torch.save(champion_net.state_dict(), gen0_path)
        champion_id = ledger.record_checkpoint(
            "gen_0", 0, str(gen0_path), sha256=compute_file_sha256(gen0_path), promoted=True, config=cfg
        )

    # 2. Replay buffer.
    train_cfg = cfg.get("training", {})
    replay_buffer = ReplayBuffer(capacity=train_cfg.get("replay_buffer_capacity", 50000))
    buffer_file = replay_dir / "replay_buffer.npz"
    if buffer_file.exists():
        log(f"Loading existing replay buffer from {buffer_file}...")
        replay_buffer.load(buffer_file)
        log(f"Replay buffer loaded with {len(replay_buffer)} positions.")

    selfplay_cfg = cfg.get("selfplay", {})
    mcts_cfg = cfg.get("mcts", {})
    arena_cfg = cfg.get("arena", {})

    games_per_gen = (
        int(games_per_generation)
        if games_per_generation is not None
        else int(selfplay_cfg.get("games_per_generation", 10))
    )
    games_per_gen = max(1, games_per_gen)
    sims = mcts_cfg.get("num_simulations", 64)
    train_steps = train_cfg.get("train_steps_per_generation", 100)
    batch_size = train_cfg.get("batch_size", 64)
    arena_games = resolve_arena_games(games_per_gen, arena_cfg)
    threshold = arena_cfg.get("win_rate_threshold", 0.55)
    log(f"Arena gating: {arena_games} games per generation (self-play batch={games_per_gen}).")

    # 3. Generations.
    for g_idx, g in enumerate(range(current_gen + 1, current_gen + 1 + generations), start=1):
        if stopped():
            break

        log(f"\n{'=' * 50}\n STARTING GENERATION {g} ({g_idx}/{generations})\n{'=' * 50}")

        champ_name = f"Champion-Gen{current_gen}"

        emit(
            "gen_start",
            generation=g,
            gen_index=g_idx,
            total_generations=generations,
            champion_name=champ_name,
            champion_gen=current_gen,
            games_per_gen=games_per_gen,
            sims=sims,
            temp_init=mcts_cfg.get("temp_init", 1.0),
            temp_final=mcts_cfg.get("temp_final", 0.0),
            temp_decay_plies=mcts_cfg.get("temp_decay_plies", 30),
            replay_buffer_size=len(replay_buffer),
            arena_num_games=arena_games,
        )

        # A. Self-play with the champion
        log(f"[Self-Play] Generating {games_per_gen} games with champion (Gen {current_gen})...")
        mcts_champ = MCTS(
            champion_net,
            c_puct=mcts_cfg.get("c_puct", 1.5),
            dirichlet_alpha=mcts_cfg.get("dirichlet_alpha", 0.3),
            dirichlet_eps=mcts_cfg.get("dirichlet_epsilon", 0.25),
            device=device,
        )

        for game_idx in range(1, games_per_gen + 1):
            if stopped(" during self-play"):
                break

            emit(
                "game_start",
                phase="selfplay",
                generation=g,
                game_idx=game_idx,
                total_games=games_per_gen,
                white=champ_name,
                black=champ_name,
            )

            pgn_file = pgns_dir / f"selfplay_gen{g}_game{game_idx}.pgn"

            def handle_move(m_data: Dict[str, Any]) -> None:
                emit("move", generation=g, game_idx=game_idx, **m_data)

            def handle_game_end(ge_data: Dict[str, Any]) -> None:
                emit(
                    "game_end",
                    generation=g,
                    game_idx=game_idx,
                    total_games=games_per_gen,
                    replay_buffer_size=len(replay_buffer) + ge_data.get("samples_count", 0),
                    **ge_data,
                )

            samples, result, plies = play_selfplay_game(
                mcts_champ,
                num_simulations=sims,
                temp_init=mcts_cfg.get("temp_init", 1.0),
                temp_final=mcts_cfg.get("temp_final", 0.0),
                temp_decay_plies=mcts_cfg.get("temp_decay_plies", 30),
                max_plies=selfplay_cfg.get("max_plies", 200),
                pgn_path=pgn_file,
                white_name=champ_name,
                black_name=champ_name,
                on_move=handle_move,
                on_game_end=handle_game_end,
                stop_event=stop_event,
            )
            if stop_event and stop_event.is_set():
                break
            replay_buffer.push_game(samples)
            ledger.record_game(
                "selfplay", champ_name, champ_name, result, plies,
                pgn_path=str(pgn_file), checkpoint_id=champion_id,
            )
            log(f"  Game {game_idx}/{games_per_gen}: {result} ({plies} plies, {len(samples)} samples)")
            progress("Self-Play", g, game_idx, games_per_gen)

        if stop_event and stop_event.is_set():
            break

        log(f"[Replay] Buffer now holds {len(replay_buffer)} positions.")
        replay_buffer.save(buffer_file)

        # B. Train a candidate copy of the champion
        log(f"[Training] Training candidate network for {train_steps} steps...")
        candidate_net = copy.deepcopy(champion_net)
        trainer = Trainer(
            candidate_net,
            learning_rate=train_cfg.get("learning_rate", 0.001),
            weight_decay=train_cfg.get("weight_decay", 0.0001),
            device=device,
        )

        for step in range(1, train_steps + 1):
            if stopped(" during optimization"):
                break

            m = trainer.train_step(replay_buffer, batch_size=batch_size)
            emit(
                "train_step",
                generation=g,
                step=step,
                total_steps=train_steps,
                loss_total=m["loss_total"],
                loss_policy=m["loss_policy"],
                loss_value=m["loss_value"],
                lr=m["lr"],
                replay_buffer_size=len(replay_buffer),
            )

            if step % max(1, train_steps // 5) == 0 or step == train_steps:
                log(f"  Step {step}/{train_steps} - loss: {m['loss_total']:.4f} "
                    f"(p: {m['loss_policy']:.4f}, v: {m['loss_value']:.4f})")
                ledger.record_train_step(
                    g, step, m["loss_total"], m["loss_policy"], m["loss_value"], m["lr"],
                    checkpoint_id=champion_id,
                )
            progress("Optimization", g, step, train_steps)

        if stop_event and stop_event.is_set():
            break

        cand_path = checkpoints_dir / f"gen_{g}.pt"
        torch.save(candidate_net.state_dict(), cand_path)
        cand_id = ledger.record_checkpoint(
            f"gen_{g}", g, str(cand_path),
            parent_id=champion_id, sha256=compute_file_sha256(cand_path), config=cfg,
        )

        # C. Arena gating
        log(f"[Arena] Gating match: Candidate Gen {g} vs Champion Gen {current_gen} ({arena_games} games)...")
        emit("arena_start", generation=g, total_games=arena_games, threshold=threshold)

        def handle_arena_game_start(gs_data: Dict[str, Any]) -> None:
            emit("game_start", generation=g, **gs_data)

        def handle_arena_move(m_data: Dict[str, Any]) -> None:
            emit("move", generation=g, **m_data)

        def handle_arena_game_end(ge_data: Dict[str, Any]) -> None:
            emit("game_end", generation=g, **ge_data)

        def handle_arena_game_complete(rec: Dict[str, Any]) -> None:
            cand_white = rec["candidate_color"] == "white"
            cand, champ = f"gen_{g}", f"gen_{current_gen}"
            ledger.record_game(
                "arena", cand if cand_white else champ, champ if cand_white else cand,
                rec["result"], rec["plies"], checkpoint_id=cand_id,
            )
            emit(
                "arena_game",
                generation=g,
                game_idx=rec["game_idx"],
                total_games=rec["total_games"],
                candidate_color=rec["candidate_color"],
                result=rec["result"],
                candidate_points=rec["candidate_points"],
                plies=rec["plies"],
                running_score=rec["running_score"],
                threshold=threshold,
            )

        promoted, win_rate, records = evaluate_candidate(
            candidate_net,
            champion_net,
            num_games=arena_games,
            num_simulations=sims,
            threshold=threshold,
            device=device,
            on_game_start=handle_arena_game_start,
            on_move=handle_arena_move,
            on_game_end=handle_arena_game_end,
            on_game_complete=handle_arena_game_complete,
            cand_name=f"gen_{g}",
            champ_name=f"gen_{current_gen}",
            stop_event=stop_event,
        )

        if stop_event and stop_event.is_set():
            log("Training stopped during arena.")
            break

        emit(
            "gen_end",
            generation=g,
            promoted=promoted,
            win_rate=win_rate,
            threshold=threshold,
            new_champion=promoted,
            champion_name=f"gen_{g}" if promoted else f"gen_{current_gen}",
        )

        log(f"[Arena Result] Candidate score: {win_rate * 100:.1f}% (threshold: {threshold * 100:.1f}%)")
        if promoted:
            log(f"🎉 Candidate Gen {g} PROMOTED to Champion!")
            ledger.promote_checkpoint(cand_id)
            champion_net, champion_id, current_gen = candidate_net, cand_id, g
            if on_progress:
                on_progress({"phase": "Promoted", "generation": g, "champion_id": cand_id, "champion_name": f"gen_{g}"})
        else:
            log(f"Candidate Gen {g} rejected. Champion remains Gen {current_gen}.")

    log("\nTraining run finished.")
