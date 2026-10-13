"""Monte Carlo Tree Search with PUCT and live analysis outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import chess
import numpy as np
import torch

from rl_chess.core.encoding import (
    ACTION_SPACE_SIZE,
    decode_action,
    encode_board,
    encode_move,
    get_legal_action_mask,
)
from rl_chess.mcts.tree import MCTSNode
from rl_chess.net.model import ChessNet, resolve_device


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
        """Advance root down the tree when a move is played."""
        if self.root is not None and self.root.is_expanded:
            action = encode_move(move, self.root.board.turn)
            if action in self.root.children:
                self.root = self.root.children[action]
                self.root.parent = None
                self.root.parent_action = None
                return
        self.root = None

    def _evaluate_leaf(
        self,
        board: chess.Board,
        history: Optional[List[chess.Board]] = None,
    ) -> Tuple[Dict[int, float], float]:
        """Evaluate leaf board using neural network; returns legal priors dict and value."""
        planes = encode_board(board, history)
        tensor = torch.from_numpy(planes).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value_t = self.net(tensor)
            logits = policy_logits.cpu().numpy()[0]
            val = float(value_t.cpu().item())

        mask = get_legal_action_mask(board)
        # Mask illegal actions with -inf
        masked_logits = np.where(mask, logits, -1e9)
        # Stable softmax over legal actions
        max_logit = np.max(masked_logits)
        exp_logits = np.exp(masked_logits - max_logit) * mask
        sum_exp = np.sum(exp_logits)

        if sum_exp > 0:
            probs = exp_logits / sum_exp
        else:
            # Fallback to uniform over legal actions
            num_legal = int(np.sum(mask))
            probs = mask.astype(np.float32) / max(1, num_legal)

        priors = {act: float(probs[act]) for act in np.flatnonzero(mask)}
        return priors, val

    def search(
        self,
        board: chess.Board,
        history: Optional[List[chess.Board]] = None,
        num_simulations: int = 100,
        temperature: float = 1.0,
        add_noise: bool = False,
    ) -> Tuple[chess.Move, np.ndarray, MCTSAnalysis]:
        """Run MCTS simulations from the current board state.
        
        Returns:
            chosen_move: Selected chess.Move
            pi: Full action probability vector of shape (4672,)
            analysis: Rich analysis structure for UI display
        """
        # Ensure root matches board
        if self.root is None or self.root.board.fen() != board.fen():
            self.root = MCTSNode(board=board.copy())

        # Expand root if not yet expanded
        if not self.root.is_expanded and not self.root.is_terminal:
            priors, _ = self._evaluate_leaf(self.root.board, history)
            legal_actions = list(priors.keys())
            self.root.expand(legal_actions, priors)

        if not self.root.legal_actions:
            # No legal moves (checkmate or stalemate)
            dummy_pi = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
            analysis = MCTSAnalysis(root_value=0.0, total_simulations=0)
            return chess.Move.null(), dummy_pi, analysis

        # Add Dirichlet noise at root for exploration in self-play
        if add_noise and self.root.legal_actions:
            num_legal = len(self.root.legal_actions)
            noise = np.random.dirichlet([self.dirichlet_alpha] * num_legal)
            for idx, a in enumerate(self.root.legal_actions):
                p = self.root.priors[a]
                self.root.priors[a] = (1.0 - self.dirichlet_eps) * p + self.dirichlet_eps * float(noise[idx])

        # Run simulations
        for _ in range(num_simulations):
            node = self.root
            search_path: List[Tuple[MCTSNode, int]] = []

            # 1. Selection
            while node.is_expanded and not node.is_terminal:
                action = node.select_action(self.c_puct)
                search_path.append((node, action))

                if action not in node.children:
                    # Need to create child
                    child_board = node.board.copy()
                    move = decode_action(action, board=node.board)
                    child_board.push(move)
                    child_node = MCTSNode(
                        board=child_board,
                        parent=node,
                        parent_action=action,
                        prior=node.priors[action],
                    )
                    node.children[action] = child_node
                    node = child_node
                    break
                else:
                    node = node.children[action]

            # 2. Evaluation / Expansion
            if node.is_terminal:
                value = node.terminal_value if node.terminal_value is not None else 0.0
            else:
                priors, value = self._evaluate_leaf(node.board, history)
                node.expand(list(priors.keys()), priors)

            # 3. Backpropagation (alternating turns)
            leaf_value = value
            for parent_node, action in reversed(search_path):
                leaf_value = -leaf_value
                parent_node.visit_counts[action] += 1
                parent_node.total_values[action] += leaf_value

        # Construct visit distribution pi
        pi = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        total_visits = sum(self.root.visit_counts.values())

        if total_visits == 0:
            # Fallback uniform over legal
            for a in self.root.legal_actions:
                pi[a] = 1.0 / len(self.root.legal_actions)
        elif temperature <= 1e-3:
            # Greedy
            best_action = max(self.root.legal_actions, key=lambda a: self.root.visit_counts[a])
            pi[best_action] = 1.0
        else:
            # Softened by temperature
            exp_visits = {a: (self.root.visit_counts[a] ** (1.0 / temperature)) for a in self.root.legal_actions}
            sum_exp = sum(exp_visits.values())
            for a, v in exp_visits.items():
                pi[a] = v / max(1e-8, sum_exp)

        # Select action according to temperature
        if temperature <= 1e-3:
            chosen_action = max(
                self.root.legal_actions,
                key=lambda a: (self.root.visit_counts[a], self.root.q_value(a)),
            )
        else:
            actions = self.root.legal_actions
            probs = np.array([pi[a] for a in actions], dtype=np.float64)
            sum_p = np.sum(probs)
            if sum_p > 0:
                probs = probs / sum_p
                # Guarantee exact sum=1.0 for numpy
                probs[-1] = max(0.0, 1.0 - np.sum(probs[:-1]))
                chosen_action = int(np.random.choice(actions, p=probs))
            else:
                chosen_action = actions[0]

        chosen_move = decode_action(chosen_action, board=self.root.board)

        # Extract analysis details
        analysis = self._extract_analysis(total_visits)

        return chosen_move, pi, analysis

    def _extract_analysis(self, total_visits: int) -> MCTSAnalysis:
        if self.root is None:
            return MCTSAnalysis(root_value=0.0, total_simulations=0)

        # Candidate moves sorted by visit count descending
        candidates: List[CandidateMoveAnalysis] = []
        for a in self.root.legal_actions:
            v = self.root.visit_counts[a]
            mv = decode_action(a, board=self.root.board)
            san = self.root.board.san(mv) if mv in self.root.board.legal_moves else str(mv)
            candidates.append(
                CandidateMoveAnalysis(
                    action=a,
                    move=mv,
                    san=san,
                    visits=v,
                    visit_share=v / max(1, total_visits),
                    prior=self.root.priors.get(a, 0.0),
                    q_value=self.root.q_value(a),
                )
            )
        candidates.sort(key=lambda c: (c.visits, c.q_value), reverse=True)

        # Root value estimate: weighted average Q-value of root children
        root_val = 0.0
        if total_visits > 0:
            root_val = sum(c.visit_share * c.q_value for c in candidates)

        # Extract principal variation (PV)
        pv_moves: List[chess.Move] = []
        curr = self.root
        while curr is not None and curr.is_expanded and not curr.is_terminal and curr.visit_counts:
            best_a = max(curr.legal_actions, key=lambda a: curr.visit_counts.get(a, 0), default=None)
            if best_a is None or curr.visit_counts.get(best_a, 0) == 0:
                break
            mv = decode_action(best_a, board=curr.board)
            pv_moves.append(mv)
            curr = curr.children.get(best_a)

        pv_board = self.root.board.copy()
        pv_sans = []
        for mv in pv_moves:
            if mv in pv_board.legal_moves:
                pv_sans.append(pv_board.san(mv))
                pv_board.push(mv)
            else:
                pv_sans.append(str(mv))

        return MCTSAnalysis(
            root_value=root_val,
            total_simulations=total_visits,
            candidates=candidates,
            principal_variation=pv_moves,
            pv_san=" ".join(pv_sans),
        )
