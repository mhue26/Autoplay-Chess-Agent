"""Replay buffer for AlphaZero chess self-play training."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self.states: List[np.ndarray] = []
        self.policies: List[np.ndarray] = []
        self.values: List[float] = []
        self.position = 0

    def push(self, state: np.ndarray, policy: np.ndarray, value: float) -> None:
        if len(self.states) < self.capacity:
            self.states.append(state)
            self.policies.append(policy)
            self.values.append(value)
        else:
            self.states[self.position] = state
            self.policies[self.position] = policy
            self.values[self.position] = value
        self.position = (self.position + 1) % self.capacity

    def push_game(self, game_samples: List[Tuple[np.ndarray, np.ndarray, float]]) -> None:
        for s, p, v in game_samples:
            self.push(s, p, v)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = np.random.choice(len(self.states), size=batch_size, replace=False)
        batch_states = np.stack([self.states[i] for i in indices])
        batch_policies = np.stack([self.policies[i] for i in indices])
        batch_values = np.array([self.values[i] for i in indices], dtype=np.float32)
        return batch_states, batch_policies, batch_values

    def __len__(self) -> int:
        return len(self.states)

    def save(self, file_path: str | Path) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path),
            states=np.array(self.states, dtype=np.float32),
            policies=np.array(self.policies, dtype=np.float32),
            values=np.array(self.values, dtype=np.float32),
        )

    def load(self, file_path: str | Path) -> None:
        path = Path(file_path)
        if not path.exists():
            return
        data = np.load(str(path))
        self.states = list(data["states"])
        self.policies = list(data["policies"])
        self.values = [float(v) for v in data["values"]]
        self.position = len(self.states) % self.capacity
