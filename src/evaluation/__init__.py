"""
Модуль оценки и тестирования.

Включает:
- ComparativeTester: Сравнительное тестирование методов
- LongTermEvaluator: Долгосрочная оценка
- ExperimentRunner: Базовые эксперименты
- Функции расчета метрик
"""

from .comparative_tester import ComparativeTester
from .long_term_evaluator import LongTermEvaluator
from .experiment_runner import ExperimentRunner
from .metrics import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_f1_at_k,
    calculate_coverage,
    calculate_diversity,
    calculate_novelty,
    calculate_cumulative_discounted_reward,
    calculate_retention_rate,
)

__all__ = [
    'ComparativeTester',
    'LongTermEvaluator',
    'ExperimentRunner',
    'calculate_precision_at_k',
    'calculate_recall_at_k',
    'calculate_f1_at_k',
    'calculate_coverage',
    'calculate_diversity',
    'calculate_novelty',
    'calculate_cumulative_discounted_reward',
    'calculate_retention_rate',
]
