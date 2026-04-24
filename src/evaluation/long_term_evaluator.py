"""
Долгосрочная оценка полезности рекомендаций.

LongTermEvaluator прогоняет DQN-агента и базовые модели на длинных
траекториях, считает CDR, Retention Rate, Learning Slope и Final Coverage,
а также короткие метрики Precision@K / Recall@K / F1@K против
внешнего ground truth (например, ``Rating > 3`` для ITM-Rec или
``Outcome >= 0.5`` для OULAD).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .metrics import (
    calculate_cumulative_discounted_reward,
    calculate_learning_slope,
    calculate_retention_rate,
)

logger = logging.getLogger("rec_sys_edu")


def _default_retention_threshold(dataset_type: str) -> float:
    """Порог ``r_t > τ_r`` по §3.5. Для ITM-Rec шаговая награда в ``[0, 1]``,
    для OULAD — инкременты прокси (обычно ``±0.05``)."""
    if dataset_type.lower() == "itmrec":
        return 0.5
    return 0.0


class LongTermEvaluator:
    """Оценка долгосрочной полезности рекомендаций.

    Метрики соответствуют §3.5 и §6 (H1):
    - Cumulative Discounted Reward (CDR)
    - Retention Rate (удержание пользователей)
    - Learning Slope (тренд улучшения)
    - Final Coverage (покрытие каталога к концу траектории)
    - Precision@K / Recall@K / F1@K vs внешний ground truth
    """

    def __init__(
        self,
        env,
        dqn_agent,
        baseline_models: Dict[str, Callable],
        *,
        dataset_type: str = "itmrec",
        retention_threshold: Optional[float] = None,
        k: int = 10,
        gamma: float = 0.99,
        ground_truth_by_user: Optional[Mapping[int, List[int]]] = None,
    ) -> None:
        """
        Args:
            env: среда ``EducationalEnvironment`` / ``OULADEnvironment``.
            dqn_agent: обученный агент.
            baseline_models: словарь ``{name: recommender_fn(user_id, context, k)}``.
            dataset_type: ``'itmrec'`` или ``'oulad'`` — влияет на пороги.
            retention_threshold: порог ``τ_r`` для Retention Rate.
                По умолчанию берётся из ``_default_retention_threshold``.
            k: отсечка Top-K для Precision/Recall.
            gamma: коэффициент дисконтирования для CDR.
            ground_truth_by_user: релевантные item_id для каждого user_id.
                Если не передан — считаем из траектории по порогу награды
                (легаси-поведение, не рекомендуется для OULAD).
        """
        self.env = env
        self.dqn_agent = dqn_agent
        self.baseline_models = baseline_models
        self.dataset_type = dataset_type.lower()
        self.retention_threshold = (
            float(retention_threshold)
            if retention_threshold is not None
            else _default_retention_threshold(self.dataset_type)
        )
        self.k = int(k)
        self.gamma = float(gamma)
        self.learning_slope_signal = (
            "cumulative_reward" if self.dataset_type == "oulad" else "reward"
        )
        self.ground_truth_by_user: Dict[int, List[int]] = {
            int(u): [int(i) for i in items]
            for u, items in (ground_truth_by_user or {}).items()
        }

    # ------------------------------------------------------------------
    # Основной цикл
    # ------------------------------------------------------------------

    def run_long_term_experiment(
        self,
        n_users: int = 30,
        trajectory_length: int = 100,
    ) -> Dict[str, Dict[str, Any]]:
        """Запуск долгосрочного эксперимента."""
        results: Dict[str, Dict[str, Any]] = {}

        unique_users = self.env.ratings["UserID_encoded"].unique()
        test_users = np.random.choice(
            unique_users,
            min(n_users, len(unique_users)),
            replace=False,
        )

        for model_name, model in [("DQN", self.dqn_agent)] + list(self.baseline_models.items()):
            logger.info("LongTerm: запускаем %s", model_name)

            user_results = []
            for user_idx in test_users:
                state = self.env.reset(user_id=int(user_idx))

                rewards: List[float] = []
                recommended_items: List[int] = []

                for _ in range(trajectory_length):
                    mask = None
                    if hasattr(self.env, "get_action_mask"):
                        try:
                            mask = self.env.get_action_mask()
                        except Exception:
                            mask = None

                    if model_name == "DQN":
                        action = model.get_action(state, epsilon=0.01, action_mask=mask)
                    else:
                        context = getattr(self.env, "current_context", {})
                        recommendations = model(int(user_idx), context, k=1)
                        action = (
                            recommendations[0]
                            if recommendations
                            else int(np.random.randint(0, self.env.dataset.n_items))
                        )

                    recommended_items.append(int(action))

                    next_state, reward, done, _info = self.env.step(int(action))
                    rewards.append(float(reward))

                    state = next_state
                    if done:
                        break

                user_metrics = self._calculate_user_metrics(
                    user_id=int(user_idx),
                    recommendations=recommended_items,
                    rewards=rewards,
                )
                user_results.append(user_metrics)

            results[model_name] = self._aggregate_results(user_results)

        return results

    # ------------------------------------------------------------------
    # Метрики на одного пользователя
    # ------------------------------------------------------------------

    def _get_relevant_items(
        self,
        user_id: int,
        recommendations: List[int],
        rewards: List[float],
    ) -> List[int]:
        """Получает множество релевантных элементов для пользователя.

        Приоритет: внешний ground truth (``Rating>3`` или ``Outcome>=0.5``).
        Fallback — шаги с ``reward > retention_threshold`` в самой
        траектории.
        """
        gt = self.ground_truth_by_user.get(int(user_id))
        if gt:
            return list(gt)
        return [
            item for item, reward in zip(recommendations, rewards)
            if reward > self.retention_threshold
        ]

    def _calculate_user_metrics(
        self,
        user_id: int,
        recommendations: List[int],
        rewards: List[float],
    ) -> Dict[str, float]:
        metrics: Dict[str, Any] = {}

        relevant_items = set(self._get_relevant_items(user_id, recommendations, rewards))

        k = self.k
        top_k = recommendations[:k]
        true_positives = len(set(top_k) & relevant_items)
        precision = true_positives / max(len(top_k), 1)
        recall = true_positives / max(len(relevant_items), 1) if relevant_items else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        metrics[f"Precision@{k}"] = float(precision)
        metrics[f"Recall@{k}"] = float(recall)
        metrics[f"F1@{k}"] = float(f1)

        metrics["CDR"] = calculate_cumulative_discounted_reward(rewards, gamma=self.gamma)
        metrics["Retention_Rate"] = calculate_retention_rate(
            rewards, threshold=self.retention_threshold
        )

        metrics["Learning_Slope"] = calculate_learning_slope(
            rewards,
            dataset_type=self.dataset_type,
            signal="auto",
        )
        if self.dataset_type == "oulad":
            metrics["Reward_Learning_Slope"] = calculate_learning_slope(
                rewards,
                dataset_type=self.dataset_type,
                signal="reward",
            )

        coverage_progress = []
        n_items = getattr(self.env.dataset, "n_items", max(len(set(recommendations)), 1))
        for i in range(1, len(recommendations) + 1):
            coverage_progress.append(
                len(set(recommendations[:i])) / min(i, n_items)
            )
        metrics["Coverage_Progress"] = coverage_progress[-1] if coverage_progress else 0.0
        metrics["Final_Coverage"] = len(set(recommendations)) / max(n_items, 1)

        metrics["rewards_progress"] = list(rewards)
        return metrics

    def _aggregate_results(self, user_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        df = pd.DataFrame(user_results)

        k = self.k
        cols = [
            "CDR", f"Precision@{k}", f"Recall@{k}", f"F1@{k}",
            "Retention_Rate", "Learning_Slope",
            "Coverage_Progress", "Final_Coverage",
        ]
        if "Reward_Learning_Slope" in df.columns:
            cols.append("Reward_Learning_Slope")
        cols = [c for c in cols if c in df.columns]

        return {
            "mean": df[cols].mean(),
            "std": df[cols].std(),
            "data": df,
        }

    # ------------------------------------------------------------------
    # Статистика
    # ------------------------------------------------------------------

    def run_significance_test(
        self,
        results: Dict[str, Dict[str, Any]],
        metric: str = "CDR",
        reference: str = "DQN",
        alpha: float = 0.05,
    ) -> Dict[str, Dict[str, float]]:
        """Welch t-test ``reference`` против остальных моделей по метрике."""
        if reference not in results:
            raise ValueError(f"Модель {reference!r} отсутствует в результатах")
        ref_values = results[reference]["data"][metric].to_numpy()
        report: Dict[str, Dict[str, float]] = {}
        for name, payload in results.items():
            if name == reference:
                continue
            other = payload["data"][metric].to_numpy()
            if len(ref_values) < 2 or len(other) < 2:
                report[name] = {
                    "t_stat": float("nan"),
                    "p_value": float("nan"),
                    "significant": False,
                    "mean_diff": float(np.mean(ref_values) - np.mean(other)) if len(other) else 0.0,
                }
                continue
            t_stat, p_value = stats.ttest_ind(ref_values, other, equal_var=False)
            report[name] = {
                "t_stat": float(t_stat),
                "p_value": float(p_value),
                "significant": bool(p_value < alpha),
                "mean_diff": float(np.mean(ref_values) - np.mean(other)),
            }
        return report

    # ------------------------------------------------------------------
    # Визуализация
    # ------------------------------------------------------------------

    def visualize_long_term_results(
        self,
        results: Dict[str, Dict[str, Any]],
        save_path: Optional[Path] = None,
        show: bool = False,
    ) -> Optional[Path]:
        """Рисует CDR / Retention / Slope / Final Coverage / Precision по моделям.

        Возвращает путь к сохраненному PNG (если ``save_path`` задан).
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        models = list(results.keys())
        k = self.k

        cdrs = [results[m]["mean"]["CDR"] for m in models]
        cdrs_std = [results[m]["std"]["CDR"] for m in models]
        axes[0, 0].bar(models, cdrs, yerr=cdrs_std, capsize=5, alpha=0.7, color="tab:blue")
        axes[0, 0].set_title("Кумулятивная дисконтированная награда (CDR)")
        axes[0, 0].set_ylabel("CDR")
        axes[0, 0].tick_params(axis="x", rotation=30)
        axes[0, 0].grid(True, alpha=0.3, axis="y")

        retentions = [results[m]["mean"]["Retention_Rate"] for m in models]
        retentions_std = [results[m]["std"]["Retention_Rate"] for m in models]
        axes[0, 1].bar(models, retentions, yerr=retentions_std, capsize=5, alpha=0.7, color="tab:green")
        axes[0, 1].set_title(f"Retention Rate (порог t_r={self.retention_threshold:.2f})")
        axes[0, 1].set_ylabel("Retention Rate")
        axes[0, 1].tick_params(axis="x", rotation=30)
        axes[0, 1].grid(True, alpha=0.3, axis="y")

        slopes = [results[m]["mean"]["Learning_Slope"] for m in models]
        slopes_std = [results[m]["std"]["Learning_Slope"] for m in models]
        axes[0, 2].bar(models, slopes, yerr=slopes_std, capsize=5, alpha=0.7, color="tab:purple")
        slope_label = (
            "Learning Slope (кумулятивная траектория)"
            if self.learning_slope_signal == "cumulative_reward"
            else "Learning Slope (тренд награды)"
        )
        axes[0, 2].set_title(slope_label)
        axes[0, 2].set_ylabel("Slope")
        axes[0, 2].axhline(y=0, color="red", linestyle="--", alpha=0.4)
        axes[0, 2].tick_params(axis="x", rotation=30)
        axes[0, 2].grid(True, alpha=0.3, axis="y")

        coverages = [results[m]["mean"]["Final_Coverage"] for m in models]
        coverages_std = [results[m]["std"]["Final_Coverage"] for m in models]
        axes[1, 0].bar(models, coverages, yerr=coverages_std, capsize=5, alpha=0.7, color="tab:orange")
        axes[1, 0].set_title("Final Coverage (покрытие каталога)")
        axes[1, 0].set_ylabel("Coverage")
        axes[1, 0].tick_params(axis="x", rotation=30)
        axes[1, 0].grid(True, alpha=0.3, axis="y")

        prec_col = f"Precision@{k}"
        precisions = [results[m]["mean"].get(prec_col, 0.0) for m in models]
        precisions_std = [results[m]["std"].get(prec_col, 0.0) for m in models]
        axes[1, 1].bar(models, precisions, yerr=precisions_std, capsize=5, alpha=0.7, color="tab:red")
        axes[1, 1].set_title(f"Precision@{k}")
        axes[1, 1].set_ylabel(f"Precision@{k}")
        axes[1, 1].tick_params(axis="x", rotation=30)
        axes[1, 1].grid(True, alpha=0.3, axis="y")

        baseline_names = [m for m in models if m != "DQN"]
        axes[1, 2].axis("off")
        axes[1, 2].set_title("Значимость CDR (Welch t-test vs DQN)")
        if baseline_names and "DQN" in results:
            lines = []
            dqn_data = results["DQN"]["data"]["CDR"]
            for name in baseline_names:
                baseline_data = results[name]["data"]["CDR"]
                if len(dqn_data) >= 2 and len(baseline_data) >= 2:
                    t_stat, p_value = stats.ttest_ind(dqn_data, baseline_data, equal_var=False)
                else:
                    t_stat, p_value = float("nan"), float("nan")
                sign = " *" if (isinstance(p_value, float) and p_value < 0.05) else ""
                lines.append(
                    f"{name}: t={t_stat:.2f}, p={p_value:.4f}{sign}"
                )
            text = "DQN vs:\n" + "\n".join(lines)
            axes[1, 2].text(0.05, 0.5, text, fontsize=10, va="center",
                             family="monospace")

        plt.tight_layout()

        path = None
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
            path = save_path
        if show:
            plt.show()
        plt.close(fig)
        return path
