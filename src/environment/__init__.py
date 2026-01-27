"""
Модуль среды обучения.

Включает:
- EducationalEnvironment: Симулятор образовательной среды
- Функции кодирования состояний
"""

from .educational_env import EducationalEnvironment
from .state_encoder import (
    encode_context,
    encode_history,
    get_demographic_vector
)

__all__ = [
    'EducationalEnvironment',
    'encode_context',
    'encode_history',
    'get_demographic_vector',
]
