"""
Модуль среды обучения.

Включает:
- EducationalEnvironment: симулятор для ITM-Rec.
- OULADEnvironment: симулятор для OULAD (mixed-step траектории).
- Функции кодирования состояний и построения action mask.
"""

from .action_mask import (
    action_availability_features,
    any_action_available,
    available_action_count,
    build_oulad_action_mask,
)
from .educational_env import EducationalEnvironment
from .oulad_env import OULADEnvironment
from .oulad_state import (
    DEFAULT_STATE_DIM,
    aggregate_history,
    build_oulad_demographics,
    encode_oulad_state,
)
from .reward import (
    DEFAULT_ITMREC_WEIGHTS,
    DEFAULT_OULAD_WEIGHTS,
    adjust_itmrec_weights,
    calculate_cosine_novelty,
    calculate_itmrec_reward,
    calculate_oulad_step_reward,
    calculate_oulad_terminal_bonus,
    calculate_reward,
    demographic_multiplier,
)
from .state_encoder import (
    encode_context,
    encode_history,
    get_demographic_vector,
)

__all__ = [
    "EducationalEnvironment",
    "OULADEnvironment",
    "build_oulad_action_mask",
    "action_availability_features",
    "any_action_available",
    "available_action_count",
    "encode_context",
    "encode_history",
    "get_demographic_vector",
    "encode_oulad_state",
    "aggregate_history",
    "build_oulad_demographics",
    "DEFAULT_STATE_DIM",
    "DEFAULT_ITMREC_WEIGHTS",
    "DEFAULT_OULAD_WEIGHTS",
    "adjust_itmrec_weights",
    "calculate_cosine_novelty",
    "calculate_itmrec_reward",
    "calculate_oulad_step_reward",
    "calculate_oulad_terminal_bonus",
    "calculate_reward",
    "demographic_multiplier",
]
