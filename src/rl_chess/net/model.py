"""Dual-headed residual network for chess policy and value evaluation."""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_chess.core.encoding import ACTION_SPACE_SIZE, NUM_INPUT_CHANNELS, NUM_MOVE_TYPES

PROFILES = {
    "tiny": {"num_blocks": 6, "channels": 64},
    "small": {"num_blocks": 10, "channels": 128},
    "medium": {"num_blocks": 15, "channels": 192},
}


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out, inplace=True)


class PolicyHead(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # Conv 1x1 projecting to NUM_MOVE_TYPES (73) planes
        self.conv = nn.Conv2d(channels, NUM_MOVE_TYPES, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(NUM_MOVE_TYPES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, channels, 8, 8)
        out = F.relu(self.bn(self.conv(x)), inplace=True)  # (B, 73, 8, 8)
        # Permute to (B, rank=8, file=8, plane_idx=73) so flattening matches from_sq * 73 + plane_idx
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(x.size(0), ACTION_SPACE_SIZE)  # (B, 4672)


class ValueHead(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, channels, 8, 8)
        out = F.relu(self.bn(self.conv(x)), inplace=True)  # (B, 1, 8, 8)
        out = out.view(x.size(0), 64)
        out = F.relu(self.fc1(out), inplace=True)
        out = torch.tanh(self.fc2(out))  # (B, 1)
        return out.squeeze(-1)  # (B,)


class ChessNet(nn.Module):
    def __init__(self, num_blocks: int = 6, channels: int = 64, in_channels: int = NUM_INPUT_CHANNELS):
        super().__init__()
        self.num_blocks = num_blocks
        self.channels = channels
        self.in_channels = in_channels

        # Stem
        self.stem = ConvBlock(in_channels, channels)

        # Residual tower
        self.res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_blocks)])

        # Dual heads
        self.policy_head = PolicyHead(channels)
        self.value_head = ValueHead(channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Tensor of shape (B, C, 8, 8)
        Returns:
            policy_logits: Tensor of shape (B, 4672)
            value: Tensor of shape (B,) in [-1, 1]
        """
        out = self.stem(x)
        for block in self.res_blocks:
            out = block(out)
        policy_logits = self.policy_head(out)
        value = self.value_head(out)
        return policy_logits, value


def resolve_device(device_str: str = "auto") -> torch.device:
    """Resolve compute device: auto prioritizes MPS (Apple Silicon), then CUDA, then CPU."""
    if device_str == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


def create_model(
    profile: str = "tiny",
    num_blocks: Optional[int] = None,
    channels: Optional[int] = None,
    device: str = "auto",
) -> ChessNet:
    """Create a ChessNet according to a named profile or custom dimensions."""
    cfg = PROFILES.get(profile, PROFILES["tiny"])
    blocks = num_blocks if num_blocks is not None else cfg["num_blocks"]
    ch = channels if channels is not None else cfg["channels"]

    net = ChessNet(num_blocks=blocks, channels=ch)
    dev = resolve_device(device)
    net.to(dev)
    return net
