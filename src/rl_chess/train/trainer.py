"""Neural network trainer on self-play experience."""

from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.nn.functional as F
import torch.optim as optim

from rl_chess.net.model import ChessNet, resolve_device
from rl_chess.train.replay import ReplayBuffer


class Trainer:
    def __init__(
        self,
        net: ChessNet,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        device: str = "auto",
    ):
        self.net = net
        self.device = resolve_device(device)
        self.net.to(self.device)

        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def train_step(self, replay_buffer: ReplayBuffer, batch_size: int = 64) -> Dict[str, float]:
        if len(replay_buffer) < batch_size:
            return {"loss_total": 0.0, "loss_policy": 0.0, "loss_value": 0.0, "lr": 0.0}

        self.net.train()
        states, target_pis, target_vals = replay_buffer.sample(batch_size)

        states_t = torch.from_numpy(states).to(self.device)
        target_pis_t = torch.from_numpy(target_pis).to(self.device)
        target_vals_t = torch.from_numpy(target_vals).to(self.device)

        self.optimizer.zero_grad()
        policy_logits, value_preds = self.net(states_t)

        # Policy loss: cross entropy with soft MCTS visit targets
        log_probs = F.log_softmax(policy_logits, dim=-1)
        loss_policy = -torch.sum(target_pis_t * log_probs) / batch_size

        # Value loss: MSE
        loss_value = F.mse_loss(value_preds, target_vals_t)

        loss_total = loss_policy + loss_value
        loss_total.backward()
        self.optimizer.step()

        current_lr = self.optimizer.param_groups[0]["lr"]
        return {
            "loss_total": float(loss_total.item()),
            "loss_policy": float(loss_policy.item()),
            "loss_value": float(loss_value.item()),
            "lr": float(current_lr),
        }
