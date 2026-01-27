"""
Модуль обучения.

Включает:
- DQNTrainer: Тренер для обучения DQN агента
- PrioritizedReplayBuffer: Приоритетный буфер воспроизведения опыта
- Конфигурации обучения
"""

from .trainer import DQNTrainer
from .replay_buffer import PrioritizedReplayBuffer
from .config import (
    TRAIN_CONFIG,
    DEEPFM_CONFIG,
    REPLAY_BUFFER_CONFIG,
    DUELING_DQN_CONFIG
)

__all__ = [
    'DQNTrainer',
    'PrioritizedReplayBuffer',
    'TRAIN_CONFIG',
    'DEEPFM_CONFIG',
    'REPLAY_BUFFER_CONFIG',
    'DUELING_DQN_CONFIG',
]
