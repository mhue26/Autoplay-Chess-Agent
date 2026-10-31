"""PUCT Monte Carlo Tree Search guided by the policy/value network."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import chess
import numpy as np
import torch

from rl_chess.encoding import (
    ACTION_SPACE_SIZE,
    decode_action,
    encode_board,
    encode_move,
    get_legal_action_mask,
)
from rl_chess.model import ChessNet, resolve_device


class MCTSNode:
    """Search node; statistics live on the edges (per legal action)."""

    def __init__(self, board: chess.Board):
        self.board = board
        self.is_expanded = False
        self.legal_actions: List[int] = []
        self.priors: Dict[int, float] = {}
        self.visit_counts: Dict[int, int] = {}
        self.total_values: Dict[int, float] = {}
        self.children: Dict[int, "MCTSNode"] = {}

        self.is_terminal = board.is_game_over()
        self.terminal_value: Optional[float] = None
        if self.is_terminal:
            outcome = board.outcome()
            # Value is from the side to move: a mated side to move has lost.
            self.terminal_value = 0.0 if outcome is None or outcome.winner is None else (
                1.0 if outcome.winner == board.turn else -1.0
            )

    def expand(self, priors: Dict[int, float]) -> None:
        self.legal_actions = list(priors)
        self.priors = priors
        self.visit_counts = {a: 0 for a in priors}
        self.total_values = {a: 0.0 for a in priors}
        self.is_expanded = True

    def q_value(self, action: int) -> float:
        n = self.visit_counts.get(action, 0)
        return self.total_values[action] / n if n else 0.0

    def select_action(self, c_puct: float) -> int:
        sqrt_total = math.sqrt(max(1, sum(self.visit_counts.values())))
        return max(
            self.legal_actions,
            key=lambda a: self.q_value(a) + c_puct * self.priors[a] * sqrt_total / (1 + self.visit_counts[a]),
        )


@dataclass
class CandidateMoveAnalysis:
    action: int
    move: chess.Move
    san: str
    visits: int
    visit_share: float
    prior: float
    q_value: float


@dataclass
class MCTSAnalysis:
    root_value: float
    total_simulations: int
    candidates: List[CandidateMoveAnalysis] = field(default_factory=list)
    principal_variation: List[chess.Move] = field(default_factory=list)
    pv_san: str = ""


class MCTS:
    def __init__(
        self,
        net: ChessNet,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        device: str = "auto",
    ):
        self.net = net
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.device = resolve_device(device)
        self.net.to(self.device)
        self.net.eval()
        self.root: Optional[MCTSNode] = None

    def reset(self) -> None:
        self.root = None

    def advance_root(self, move: chess.Move) -> None:
        """Reuse the subtree under ``move`` as the new root when it exists."""
        if self.root is not None and self.root.is_expanded:
            self.root = self.root.children.get(encode_move(move, self.root.board.turn))
        else:
            self.root = None

    def _evaluate(self, board: chess.Board, history: Optional[List[chess.Board]]) -> Tuple[Dict[int, float], float]:
        """Network priors (softmax over legal actions) and value for the side to move."""
        planes = torch.from_numpy(encode_board(board, history)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.net(planes)
        logits = logits.cpu().numpy()[0]

        legal = np.flatnonzero(get_legal_action_mask(board))
        legal_logits = logits[legal]
        probs = np.exp(legal_logits - legal_logits.max())
        probs /= probs.sum()
        return {int(a): float(p) for a, p in zip(legal, probs)}, float(value.item())

    def search(
        self,
        board: chess.Board,
        history: Optional[List[chess.Board]] = None,
        num_simulations: int = 100,
        temperature: float = 1.0,
        add_noise: bool = False,
    ) -> Tuple[chess.Move, np.ndarray, MCTSAnalysis]:
        """Run MCTS from ``board``.

        Returns the chosen move, the action probability vector pi (4672,), and UI analysis.
        With no legal moves, returns a null move, a zero pi, and an empty analysis.
        """
        if self.root is None or self.root.board.fen() != board.fen():
            self.root = MCTSNode(board.copy())
        root = self.root

        if not root.is_expanded and not root.is_terminal:
            root.expand(self._evaluate(root.board, history)[0])

        if not root.legal_actions:
            return chess.Move.null(), np.zeros(ACTION_SPACE_SIZE, dtype=np.float32), MCTSAnalysis(0.0, 0)

        if add_noise:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(root.legal_actions))
            for a, n in zip(root.legal_actions, noise):
                root.priors[a] = (1.0 - self.dirichlet_eps) * root.priors[a] + self.dirichlet_eps * float(n)

        for _ in range(num_simulations):
            self._simulate(root, history)

        actions = root.legal_actions
        visits = np.array([root.visit_counts[a] for a in actions], dtype=np.float64)
        total_visits = int(visits.sum())

        if temperature <= 1e-3:
            chosen = max(range(len(actions)), key=lambda i: (visits[i], root.q_value(actions[i])))
            probs = np.zeros(len(actions))
            probs[chosen] = 1.0
        else:
            probs = visits ** (1.0 / temperature) if total_visits else np.ones(len(actions))
            probs /= probs.sum()
            chosen = int(np.random.choice(len(actions), p=probs))

        pi = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        pi[actions] = probs
        return decode_action(actions[chosen], root.board), pi, self._analysis(total_visits)

    def _simulate(self, root: MCTSNode, history: Optional[List[chess.Board]]) -> None:
        """One select -> expand/evaluate -> backup pass."""
        node = root
        path: List[Tuple[MCTSNode, int]] = []

        while node.is_expanded and not node.is_terminal:
            action = node.select_action(self.c_puct)
            path.append((node, action))
            if action not in node.children:
                child_board = node.board.copy()
                child_board.push(decode_action(action, node.board))
                node.children[action] = MCTSNode(child_board)
                node = node.children[action]
                break
            node = node.children[action]

        if node.is_terminal:
            value = node.terminal_value
        else:
            priors, value = self._evaluate(node.board, history)
            node.expand(priors)

        # Values alternate perspective with each ply.
        for parent, action in reversed(path):
            value = -value
            parent.visit_counts[action] += 1
            parent.total_values[action] += value

    def _analysis(self, total_visits: int) -> MCTSAnalysis:
        root = self.root
        candidates = [
            CandidateMoveAnalysis(
                action=a,
                move=(move := decode_action(a, root.board)),
                san=root.board.san(move),
                visits=root.visit_counts[a],
                visit_share=root.visit_counts[a] / max(1, total_visits),
                prior=root.priors[a],
                q_value=root.q_value(a),
            )
            for a in root.legal_actions
        ]
        candidates.sort(key=lambda c: (c.visits, c.q_value), reverse=True)
        root_value = sum(c.visit_share * c.q_value for c in candidates) if total_visits else 0.0

        pv_moves: List[chess.Move] = []
        curr = root
        while curr is not None and curr.is_expanded and not curr.is_terminal and curr.visit_counts:
            best_a = max(curr.legal_actions, key=lambda a: curr.visit_counts.get(a, 0), default=None)
            if best_a is None or curr.visit_counts.get(best_a, 0) == 0:
                break
            mv = decode_action(best_a, board=curr.board)
            pv_moves.append(mv)
            curr = curr.children.get(best_a)

        pv_board = root.board.copy()
        pv_sans = []
        for mv in pv_moves:
            if mv in pv_board.legal_moves:
                pv_sans.append(pv_board.san(mv))
                pv_board.push(mv)
            else:
                pv_sans.append(str(mv))

        return MCTSAnalysis(
            root_value=root_value,
            total_simulations=total_visits,
            candidates=candidates,
            principal_variation=pv_moves,
            pv_san=" ".join(pv_sans),
        )
