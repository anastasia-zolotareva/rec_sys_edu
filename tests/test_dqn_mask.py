"""Тесты корректности action-mask в DuelingDQN (§11.9 ТЗ)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.dueling_dqn import DuelingDQN
from src.training.replay_buffer import PrioritizedReplayBuffer
from src.training.trainer import DQNTrainer


def _make_agent(state_dim: int = 16, action_dim: int = 8) -> DuelingDQN:
    torch.manual_seed(0)
    return DuelingDQN(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[32, 16],
        device=torch.device("cpu"),
    )


def test_get_action_respects_mask() -> None:
    agent = _make_agent(action_dim=8)
    state = np.zeros(16, dtype=np.float32)
    mask = np.zeros(8, dtype=np.float32)
    mask[[2, 5]] = 1.0
    observed = set()
    for _ in range(200):
        a = agent.get_action(state, epsilon=1.0, action_mask=mask)
        observed.add(a)
    assert observed <= {2, 5}


def test_get_action_fallback_when_mask_all_zero() -> None:
    agent = _make_agent(action_dim=8)
    state = np.zeros(16, dtype=np.float32)
    mask = np.zeros(8, dtype=np.float32)
    # Должен вернуть случайное действие, а не упасть.
    a = agent.get_action(state, epsilon=0.0, action_mask=mask)
    assert 0 <= a < 8


def test_get_action_greedy_argmax_under_mask() -> None:
    agent = _make_agent(action_dim=4)
    state = np.zeros(16, dtype=np.float32)
    mask = np.array([1, 0, 1, 0], dtype=np.float32)
    # Проверяем, что возвращённое действие обязательно разрешено маской
    # при жадном выборе (epsilon=0).
    a = agent.get_action(state, epsilon=0.0, action_mask=mask)
    assert mask[a] == 1.0


class _DummyEnv:
    def __init__(self, state_dim: int = 16, action_dim: int = 4) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reset_calls = 0
        self.step_calls = 0
        self.dataset = type("Dataset", (), {"n_items": action_dim})()

    def reset(self):
        self.reset_calls += 1
        return np.zeros(self.state_dim, dtype=np.float32)

    def get_action_mask(self):
        return np.ones(self.action_dim, dtype=np.float32)

    def step(self, action: int):
        self.step_calls += 1
        next_state = np.full(self.state_dim, self.step_calls, dtype=np.float32)
        return next_state, 0.1, False, {}


class _FixedQNetwork(nn.Module):
    def __init__(self, q_values):
        super().__init__()
        self.register_buffer("q_values", torch.tensor(q_values, dtype=torch.float32))

    def forward(self, states):
        batch = states.shape[0]
        return self.q_values.unsqueeze(0).expand(batch, -1)


def _make_trainer(*, max_steps_per_episode: int = 3) -> DQNTrainer:
    env = _DummyEnv()
    agent = _make_agent(state_dim=16, action_dim=4)
    buffer = PrioritizedReplayBuffer(capacity=32)
    return DQNTrainer(
        env,
        agent,
        buffer,
        {
            "gamma": 0.9,
            "lr": 1e-3,
            "batch_size": 2,
            "max_steps_per_episode": max_steps_per_episode,
            "use_action_mask": True,
        },
    )


def test_replay_buffer_returns_next_action_masks() -> None:
    buffer = PrioritizedReplayBuffer(capacity=8)
    state = np.zeros(4, dtype=np.float32)
    next_state = np.ones(4, dtype=np.float32)
    mask = np.array([1, 0, 1], dtype=np.float32)
    for _ in range(3):
        buffer.push(state, 0, 0.5, next_state, False, next_action_mask=mask)

    batch = buffer.sample(batch_size=2)
    assert batch is not None
    _, _, _, _, _, next_masks, _, _ = batch
    assert next_masks is not None
    assert next_masks.shape == (2, 3)
    assert torch.all(next_masks[:, 1] == 0)


def test_compute_td_targets_respects_mask() -> None:
    trainer = _make_trainer()
    trainer.target_network = _FixedQNetwork([1.0, 5.0, 3.0, -2.0])

    rewards = torch.tensor([0.2, 0.2], dtype=torch.float32)
    next_states = torch.zeros((2, 16), dtype=torch.float32)
    dones = torch.zeros(2, dtype=torch.float32)
    next_masks = torch.tensor(
        [[1, 0, 1, 0], [0, 0, 1, 1]],
        dtype=torch.float32,
    )

    td_targets = trainer.compute_td_targets(
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        next_action_masks=next_masks,
    )

    expected = torch.tensor(
        [0.2 + 0.9 * 3.0, 0.2 + 0.9 * 3.0],
        dtype=torch.float32,
    )
    assert torch.allclose(td_targets, expected)


def test_train_episode_uses_configured_max_steps() -> None:
    trainer = _make_trainer(max_steps_per_episode=3)

    episode_reward, _avg_loss = trainer.train_episode()

    assert trainer.env.step_calls == 3
    assert episode_reward == pytest.approx(0.3, rel=1e-6)
