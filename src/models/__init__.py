"""
Модуль моделей.

Включает:
- DeepFMSVDPlusPlus: Гибридная модель DeepFM + SVD++
- DuelingDQN: RL агент с Dueling архитектурой
"""

from .deepfm_svdpp import DeepFMSVDPlusPlus
from .dueling_dqn import DuelingDQN

__all__ = [
    'DeepFMSVDPlusPlus',
    'DuelingDQN',
]
