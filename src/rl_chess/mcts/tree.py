"""MCTS Node and Tree representation for AlphaZero chess."""

from __future__ import annotations

import math
from typing import Dict, List, Optional
import chess
import numpy as np


class MCTSNode:
    def __init__(
        self,
        board: chess.Board,
        parent: Optional[MCTSNode] = None,
        parent_action: Optional[int] = None,
        prior: float = 0.0,
    ):
        self.board = board
        self.parent = parent
        self.parent_action = parent_action
        self.prior = prior

        self.is_expanded: bool = False
        self.is_terminal: bool = False
        self.terminal_value: Optional[float] = None

        # Legal actions at this node
        self.legal_actions: List[int] = []

        # Statistics per legal action
        self.priors: Dict[int, float] = {}
        self.visit_counts: Dict[int, int] = {}
        self.total_values: Dict[int, float] = {}
        self.children: Dict[int, MCTSNode] = {}

        self._check_terminal()

    def _check_terminal(self) -> None:
        if self.board.is_game_over():
            self.is_terminal = True
            outcome = self.board.outcome()
            if outcome is None or outcome.winner is None:
                self.terminal_value = 0.0  # Draw
            elif outcome.winner == self.board.turn:
                self.terminal_value = 1.0  # Current player won (rare at node start)
            else:
                self.terminal_value = -1.0  # Current player checkmated
        else:
            self.is_terminal = False
            self.terminal_value = None

    def expand(self, legal_actions: List[int], priors: Dict[int, float]) -> None:
        """Expand node with legal actions and neural net priors."""
        self.legal_actions = legal_actions
        self.priors = priors
        for a in legal_actions:
            self.visit_counts[a] = 0
            self.total_values[a] = 0.0
        self.is_expanded = True

    def q_value(self, action: int) -> float:
        n = self.visit_counts.get(action, 0)
        if n == 0:
            return 0.0
        return self.total_values[action] / n

    def select_action(self, c_puct: float) -> int:
        """Select action maximizing PUCT formula."""
        total_visits = sum(self.visit_counts.values())
        sqrt_total = math.sqrt(max(1, total_visits))

        best_score = -float("inf")
        best_action = self.legal_actions[0]

        for a in self.legal_actions:
            n = self.visit_counts[a]
            q = self.q_value(a)
            p = self.priors.get(a, 0.0)
            u = c_puct * p * sqrt_total / (1 + n)
            score = q + u

            if score > best_score:
                best_score = score
                best_action = a

        return best_action
