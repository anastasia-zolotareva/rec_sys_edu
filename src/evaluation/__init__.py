"""
Модуль оценки и тестирования.

Включает:
- ComparativeTester: Сравнительное тестирование методов
- LongTermEvaluator: Долгосрочная оценка
- ExperimentRunner: Базовые эксперименты
- AdaptabilityAnalyzer: Анализ адаптивности (H1)
- NoveltyAblationRunner: Ablation-эксперименты по новизне
- trajectory_visualizer: визуализация траекторий
- statistics: статистические тесты
- Функции расчета метрик
"""

from .adaptability import AdaptabilityAnalyzer, AdaptabilityResult
from .comparative_tester import ComparativeTester
from .experiment_runner import ExperimentRunner
from .long_term_evaluator import LongTermEvaluator
from .metrics import (
    calculate_adaptability_score,
    calculate_coverage,
    calculate_cumulative_discounted_reward,
    calculate_diversity,
    calculate_f1_at_k,
    calculate_learning_slope,
    calculate_novelty,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_retention_rate,
    calculate_stability,
    evaluate_recommendation_set,
)
from .novelty_ablation import AblationResult, NoveltyAblationRunner
# Alias для совместимости
NoveltyAblationTester = NoveltyAblationRunner

from .statistics import (
    bootstrap_ci,
    cohens_d,
    compare_against_reference,
    welch_test,
    wilcoxon_test,
)
from .trajectory_visualizer import (
    dump_trajectories_csv,
    plot_coverage_and_novelty,
    plot_reward_trajectories,
)


# Простой класс-обертка для TrajectoryVisualizer
class TrajectoryVisualizer:
    """Визуализатор траекторий.
    
    Обертка над функциями plot_reward_trajectories, plot_coverage_and_novelty, dump_trajectories_csv.
    """
    
    def __init__(self, trajectories: dict):
        """
        Args:
            trajectories: Словарь траекторий {user_id: trajectory_data}
        """
        self.trajectories = trajectories
    
    def plot_reward_trajectories(self, output_dir):
        """Визуализирует траектории наград."""
        return plot_reward_trajectories(self.trajectories, output_dir)
    
    def plot_coverage_and_novelty(self, bundle, output_dir):
        """Визуализирует покрытие и новизну."""
        return plot_coverage_and_novelty(self.trajectories, bundle, output_dir)
    
    def dump_trajectories_csv(self, output_dir):
        """Сохраняет траектории в CSV."""
        return dump_trajectories_csv(self.trajectories, output_dir)

__all__ = [
    "AdaptabilityAnalyzer",
    "AdaptabilityResult",
    "AblationResult",
    "ComparativeTester",
    "ExperimentRunner",
    "LongTermEvaluator",
    "NoveltyAblationRunner",
    "NoveltyAblationTester",  # Alias для совместимости
    "TrajectoryVisualizer",    # Класс для визуализации траекторий
    "bootstrap_ci",
    "calculate_adaptability_score",
    "calculate_coverage",
    "calculate_cumulative_discounted_reward",
    "calculate_diversity",
    "calculate_f1_at_k",
    "calculate_learning_slope",
    "calculate_novelty",
    "calculate_precision_at_k",
    "calculate_recall_at_k",
    "calculate_retention_rate",
    "calculate_stability",
    "cohens_d",
    "compare_against_reference",
    "dump_trajectories_csv",
    "evaluate_recommendation_set",
    "plot_coverage_and_novelty",
    "plot_reward_trajectories",
    "welch_test",
    "wilcoxon_test",
]
