# RL-Chess-Agent

An AlphaZero-style chess reinforcement learning engine and local desktop application written from scratch in PyTorch, featuring:
- **Zero human heuristics**: Learns purely via tabula rasa self-play and PUCT Monte Carlo Tree Search (MCTS).
- **Apple Silicon MPS GPU accelerated**: Optimized for local Mac training with CPU and CUDA fallbacks.
- **Pygame Desktop GUI**: Real-time interactive board with threaded non-blocking agent search, live win-expectancy evaluation bar, and top candidate move arrows.
- **Calibrated Stockfish Benchmark**: Gauntlet against Stockfish across calibrated strength levels with Bradley-Terry Maximum Likelihood Elo fitting and 95% confidence intervals.
- **SQLite Universal Training Ledger**: Audits all self-play games, training steps, arena promotions, and human matches in `data/ledger.db`.

---

## Architecture

```
                          ┌─────────────────────────────────────┐
                          │         Pygame Desktop GUI          │
                          │  (Threaded MCTS, Live Eval, Arrows) │
                          └──────────────────┬──────────────────┘
                                             │
                       ┌─────────────────────┴────────────────────┐
                       ▼                                          ▼
        ┌──────────────────────────────┐          ┌──────────────────────────────┐
        │      Training Pipeline       │          │   Stockfish Elo Benchmark    │
        │ • Self-Play (PUCT + Noise)   │          │ • Gauntlet vs Calibrated SF  │
        │ • Replay Buffer     │          │ • Bradley-Terry MLE Solver   │
        │ • SGD Dual-Head ResNet       │          │ • 95% Confidence Intervals   │
        │ • 55% Arena Promotion Gate   │          │                              │
        └──────────────┬───────────────┘          └──────────────┬───────────────┘
                       │                                         │
                       └─────────────────────┬───────────────────┘
                                             ▼
                          ┌─────────────────────────────────────┐
                          │   SQLite Universal Ledger (WAL)     │
                          │  data/ledger.db & Checkpoints (.pt) │
                          └─────────────────────────────────────┘
```

### AlphaZero Encoding
- **Input Planes**: 63 channels (8×8): 4 history steps × 12 piece channels + 8 repetition channels + 7 global channels (color, turn counter, castling rights, halfmove clock).
- **Policy Space**: 8×8×73 = 4,672 discrete actions (56 queen-like moves, 8 knight jumps, 9 underpromotions to Knight/Bishop/Rook).

---

## Quickstart

### 1. Environment Setup
The project uses Python 3.11+ in a local virtual environment:

```bash
cd RL-Chess-Agent
source .venv/bin/activate
pip install -e .
```

### 2. Launch the Unified Desktop App
Launch the complete all-in-one Pygame dashboard:

```bash
rl-chess
```

The window is 1040×620 points and renders at the display's native pixel density (2× on Retina) automatically.

Everything can be controlled from within the desktop window across 4 integrated tabs:
1. **Play Computer**: Interactive chessboard, live win expectancy bar, and candidate move arrows.
2. **Training**: Choose how many generations to run, start/stop training, and watch live logs.
3. **Benchmark Elo**: Select checkpoint, run Stockfish gauntlet, and get fitted Elo ratings with 95% confidence intervals.
4. **Statistics**: Inspect total games, training steps, registered checkpoints, and complete Elo rating history.

### Optional CLI Commands (Headless / Scripts)
You can still run individual operations directly from the command line:
- `rl-chess train --generations 2`
- `rl-chess benchmark --games 10`
- `rl-chess stats`

Settings (network size, MCTS sims, games per generation, etc.) live in `configs/config.yaml`.

### Layout
```
src/rl_chess/
  encoding.py   board/move planes      model.py      dual-head ResNet
  mcts.py       PUCT search            game.py       shared game loop
  ledger.py     SQLite ledger          config.py     YAML loading
  elo.py        Bradley-Terry fit      benchmark.py  Stockfish gauntlet
  train/        self-play, arena, replay buffer, trainer, generation loop
  gui/          Pygame app, renderer, widgets, play session, workers
  cli.py        `rl-chess` entry point
```

---

## Test Suite

Run the full automated test suite (covering encoding round-trips, ResNet gradients, MCTS search, selfplay training, Bradley-Terry Elo solver, and headless Pygame rendering):

```bash
pytest tests/
```
