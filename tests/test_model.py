"""Tests for ChessNet neural network architectures and profiles."""

from __future__ import annotations

import pytest
import torch

from rl_chess.core.encoding import ACTION_SPACE_SIZE, NUM_INPUT_CHANNELS
from rl_chess.net.model import ChessNet, create_model, resolve_device


def test_model_shapes_cpu():
    model = create_model("tiny", device="cpu")
    model.eval()

    batch_size = 4
    dummy_input = torch.randn(batch_size, NUM_INPUT_CHANNELS, 8, 8, device="cpu")

    with torch.no_grad():
        policy_logits, value = model(dummy_input)

    assert policy_logits.shape == (batch_size, ACTION_SPACE_SIZE)
    assert value.shape == (batch_size,)
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0)


def test_model_backprop():
    model = create_model("tiny", device="cpu")
    model.train()

    dummy_input = torch.randn(2, NUM_INPUT_CHANNELS, 8, 8, device="cpu")
    policy_logits, value = model(dummy_input)

    target_policy = torch.randn(2, ACTION_SPACE_SIZE)
    target_value = torch.tensor([0.5, -0.5])

    loss_p = torch.mean((policy_logits - target_policy) ** 2)
    loss_v = torch.mean((value - target_value) ** 2)
    loss = loss_p + loss_v

    loss.backward()

    # Verify gradients computed for stem and heads
    assert model.stem.conv.weight.grad is not None
    assert model.policy_head.conv.weight.grad is not None
    assert model.value_head.fc2.weight.grad is not None


def test_profiles():
    net_tiny = create_model("tiny", device="cpu")
    assert net_tiny.num_blocks == 6
    assert net_tiny.channels == 64

    net_small = create_model("small", device="cpu")
    assert net_small.num_blocks == 10
    assert net_small.channels == 128


def test_mps_execution_if_available():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available on this host")

    model = create_model("tiny", device="mps")
    dummy_input = torch.randn(2, NUM_INPUT_CHANNELS, 8, 8, device="mps")

    policy_logits, value = model(dummy_input)
    assert policy_logits.device.type == "mps"
    assert value.device.type == "mps"
    assert policy_logits.shape == (2, ACTION_SPACE_SIZE)
    assert value.shape == (2,)
