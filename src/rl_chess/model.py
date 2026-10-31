"""Dual-headed residual network for chess policy and value evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_chess.encoding import ACTION_SPACE_SIZE, NUM_INPUT_CHANNELS, NUM_MOVE_TYPES


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + x, inplace=True)


class PolicyHead(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, NUM_MOVE_TYPES, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(NUM_MOVE_TYPES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn(self.conv(x)), inplace=True)  # (B, 73, 8, 8)
        # (B, rank, file, plane) so the flattened index is from_square * 73 + plane
        return out.permute(0, 2, 3, 1).reshape(x.size(0), ACTION_SPACE_SIZE)


class ValueHead(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn(self.conv(x)), inplace=True).view(x.size(0), 64)
        out = F.relu(self.fc1(out), inplace=True)
        return torch.tanh(self.fc2(out)).squeeze(-1)


class ChessNet(nn.Module):
    def __init__(self, num_blocks: int = 10, channels: int = 128):
        super().__init__()
        self.num_blocks = num_blocks
        self.channels = channels
        self.stem = ConvBlock(NUM_INPUT_CHANNELS, channels)
        self.res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_blocks)])
        self.policy_head = PolicyHead(channels)
        self.value_head = ValueHead(channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (policy_logits of shape (B, 4672), value of shape (B,) in [-1, 1])."""
        out = self.stem(x)
        for block in self.res_blocks:
            out = block(out)
        return self.policy_head(out), self.value_head(out)


def resolve_device(device_str: str = "auto") -> torch.device:
    """'auto' prefers MPS (Apple Silicon), then CUDA, then CPU."""
    if device_str == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


def create_model(num_blocks: int = 10, channels: int = 128, device: str = "auto") -> ChessNet:
    return ChessNet(num_blocks, channels).to(resolve_device(device))


def model_from_config(cfg: Dict[str, Any]) -> ChessNet:
    net_cfg = cfg.get("network", {})
    return create_model(net_cfg.get("num_blocks", 10), net_cfg.get("channels", 128), cfg.get("device", "auto"))


def load_weights(net: ChessNet, path: str | Path) -> None:
    net.load_state_dict(torch.load(path, map_location=next(net.parameters()).device))
    net.eval()
