"""
AdaptabilityAnalyzer (гипотеза H2).
Проверяет способность агента поддерживать стабильное качество рекомендаций
при смене контекста/профиля пользователя:

- ``σ_P = std({P@K(c_m)})`` и ``σ_R = std({R@K(c_m)})`` — разброс точности
  и полноты по стратам контекста/демографии.
- ``AdaptabilityScore = 1 - (σ_P + σ_R) / 2`` — итоговая устойчивость.

Для ITM-Rec страты — ``Class_encoded``, ``Semester_encoded`` и ``Lockdown_encoded``
(контекстные колонки в ``bundle.ratings``). Для OULAD — демографические
колонки пользователей (``Gender_encoded``, ``AgeBand_encoded`` и т.п.),
задаваемые снаружи.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .metrics import (
    calculate_adaptability_score,
    calculate_learning_slope,
    calculate_stability,
)

logger = logging.getLogger("rec_sys_edu")


@dataclass
class AdaptabilityResult:
    adaptability_score: float
    learning_slope: float
    precision_stability: float
    recall_stability: float
    per_user_precision: float = 0.0
    per_user_recall: float = 0.0
    by_strata: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "adaptability_score": self.adaptability_score,
            "learning_slope": self.learning_slope,
            "precision_stability": self.precision_stability,
            "recall_stability": self.recall_stability,
            "per_user_precision": self.per_user_precision,
            "per_user_recall": self.per_user_recall,
            "by_strata": self.by_strata,
        }


class AdaptabilityAnalyzer:
    """Выполняет анализ адаптивности по траекториям агента."""

    def __init__(
        self,
        env,
        agent,
        strata_columns: Optional[Sequence[str]] = None,
    ) -> None:
        self.env = env
        self.agent = agent
        self.strata_columns = list(strata_columns or [])
        bundle = getattr(env, "bundle", None)
        self.dataset_type = getattr(bundle, "dataset_type", "itmrec")

    # ------------------------------------------------------------------
    # Сбор траекторий
    # ------------------------------------------------------------------

    def collect_trajectories(
        self,
        user_ids: Iterable[int],
        trajectory_length: int = 50,
        action_mask_fn: Optional[Callable[[], np.ndarray]] = None,
    ) -> List[Dict[str, Any]]:
        """Прогоняет агента на указанных пользователях и собирает траектории.

        Возвращает список словарей ``{user_id, rewards, actions}``.
        """
        collected: List[Dict[str, Any]] = []
        for user_id in user_ids:
            state = self.env.reset(user_id=int(user_id))
            rewards: List[float] = []
            actions: List[int] = []
            for _ in range(trajectory_length):
                mask = None
                if hasattr(self.env, "get_action_mask"):
                    try:
                        mask = self.env.get_action_mask()
                    except Exception:
                        mask = None
                action = self.agent.get_action(state, epsilon=0.01, action_mask=mask)
                next_state, reward, done, _ = self.env.step(action)
                rewards.append(float(reward))
                actions.append(int(action))
                state = next_state
                if done:
                    break
            collected.append({
                "user_id": int(user_id),
                "rewards": rewards,
                "actions": actions,
            })
        return collected

    # ------------------------------------------------------------------
    # Анализ
    # ------------------------------------------------------------------

    @staticmethod
    def _precision_recall(actions: Sequence[int], ground_truth: Sequence[int], k: int):
        recs = list(actions)[:k]
        gt = set(int(x) for x in ground_truth)
        if not recs:
            return 0.0, 0.0
        tp = len(set(recs) & gt)
        precision = tp / max(len(recs), 1)
        recall = tp / max(len(gt), 1) if gt else 0.0
        return float(precision), float(recall)

    def analyze(
        self,
        trajectories: Sequence[Mapping[str, Any]],
        ground_truth_by_user: Mapping[int, Sequence[int]],
        users_df: Optional[pd.DataFrame] = None,
        ratings_df: Optional[pd.DataFrame] = None,
        k: int = 10,
    ) -> AdaptabilityResult:
        """Считает AdaptabilityScore, σ_P/σ_R и страты.

        Args:
            trajectories: список результатов ``collect_trajectories``.
            ground_truth_by_user: релевантные ``item_id`` для каждого
                ``user_id``.
            users_df: DataFrame пользователей (для демографических страт).
                Индекс/колонка ``UserID_encoded`` используется как ключ.
            ratings_df: DataFrame рейтингов (для контекстных страт
                ITM-Rec: Class/Semester/Lockdown).
            k: отсечка Top-K.
        """
        per_user_rewards = [t.get("rewards", []) for t in trajectories]
        slopes = [
            calculate_learning_slope(
                r,
                dataset_type=self.dataset_type,
                signal="auto",
            )
            for r in per_user_rewards
            if len(r) >= 2
        ]
        learning_slope = float(np.mean(slopes)) if slopes else 0.0

        # Per-user precision / recall для отчётности.
        user_records: List[Dict[str, Any]] = []
        for traj in trajectories:
            uid = int(traj["user_id"])
            gt = ground_truth_by_user.get(uid, [])
            precision, recall = self._precision_recall(traj.get("actions", []), gt, k)
            user_records.append({
                "user_id": uid,
                "precision": precision,
                "recall": recall,
                "mean_reward": float(np.mean(traj["rewards"])) if traj["rewards"] else 0.0,
                "episode_len": len(traj.get("rewards", [])),
            })
        per_user_df = pd.DataFrame(user_records)

        per_user_precision = float(per_user_df["precision"].mean()) if not per_user_df.empty else 0.0
        per_user_recall = float(per_user_df["recall"].mean()) if not per_user_df.empty else 0.0

        # --- Страты ---
        strata_values: Dict[str, Dict[str, Dict[str, float]]] = {}
        strata_precisions: List[float] = []
        strata_recalls: List[float] = []

        if not per_user_df.empty and self.strata_columns:
            # Источник атрибутов пользователя: сначала ratings (контекст),
            # потом users (демография).
            per_user_df = per_user_df.copy()

            for col in self.strata_columns:
                mapping: Optional[pd.Series] = None
                if ratings_df is not None and col in ratings_df.columns and "UserID_encoded" in ratings_df.columns:
                    # Mode value per user — наиболее характерный контекст.
                    agg = ratings_df.groupby("UserID_encoded")[col].agg(
                        lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
                    )
                    mapping = agg
                elif users_df is not None and col in users_df.columns:
                    if "UserID_encoded" in users_df.columns:
                        mapping = users_df.set_index("UserID_encoded")[col]
                    else:
                        mapping = users_df[col]
                if mapping is None:
                    continue
                values = per_user_df["user_id"].map(mapping)
                per_user_df[col] = values

                grouped = per_user_df.dropna(subset=[col]).groupby(col)
                strata_table: Dict[str, Dict[str, float]] = {}
                col_precisions: List[float] = []
                col_recalls: List[float] = []
                for stratum_value, sub in grouped:
                    mean_p = float(sub["precision"].mean())
                    mean_r = float(sub["recall"].mean())
                    mean_rw = float(sub["mean_reward"].mean())
                    strata_table[str(stratum_value)] = {
                        "precision": mean_p,
                        "recall": mean_r,
                        "mean_reward": mean_rw,
                        "n_users": int(len(sub)),
                    }
                    col_precisions.append(mean_p)
                    col_recalls.append(mean_r)
                if strata_table:
                    strata_values[col] = strata_table
                    strata_precisions.extend(col_precisions)
                    strata_recalls.extend(col_recalls)

        # Если страт не набралось (нет колонок или нет совпадений) — опускаемся
        # до per-user дисперсии, чтобы метрика всё равно была определена.
        if strata_precisions and strata_recalls and len(strata_precisions) >= 2:
            precisions_for_sigma = strata_precisions
            recalls_for_sigma = strata_recalls
        else:
            precisions_for_sigma = per_user_df["precision"].tolist() if not per_user_df.empty else []
            recalls_for_sigma = per_user_df["recall"].tolist() if not per_user_df.empty else []

        precision_stability = calculate_stability(precisions_for_sigma)
        recall_stability = calculate_stability(recalls_for_sigma)
        adaptability_score = calculate_adaptability_score(
            precisions_for_sigma, recalls_for_sigma
        )

        return AdaptabilityResult(
            adaptability_score=adaptability_score,
            learning_slope=learning_slope,
            precision_stability=precision_stability,
            recall_stability=recall_stability,
            per_user_precision=per_user_precision,
            per_user_recall=per_user_recall,
            by_strata=strata_values,
        )
