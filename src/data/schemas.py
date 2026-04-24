"""
Общие структуры данных для ITM-Rec и OULAD.

`DatasetBundle` — унифицированный контейнер, возвращаемый функциями
`build_itmrec_bundle()` и `build_oulad_bundle()`. Он используется средой,
моделями и тестовыми контурами, так что код выше по стеку не зависит
от конкретного датасета.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class DatasetBundle:
    """Контейнер с подготовленными данными для обучения и тестирования.

    Обязательные поля используются везде. Поля-списки имеют значения
    по умолчанию, чтобы можно было опускать неактуальные для конкретного
    датасета элементы.
    """

    # Тип датасета: "itmrec" | "oulad"
    dataset_type: str

    # Основная "ratings/ratings-подобная" таблица с закодированными полями
    # (UserID_encoded / ItemID_encoded и, при необходимости, Context_*_encoded)
    ratings: pd.DataFrame

    # Демография / профили пользователей (studentInfo для OULAD)
    users: pd.DataFrame

    # Каталог рекомендуемых объектов (items для ITM-Rec, step catalog для OULAD)
    items: pd.DataFrame

    # Целевые колонки для мультикритериальной модели
    # ITM-Rec: ["Rating", "App", "Data", "Ease"]
    # OULAD:   ["Mastery", "Engagement", "SelfRegulation", "Outcome"]
    target_columns: List[str] = field(default_factory=list)

    # Размерности
    n_users: int = 0
    n_items: int = 0
    state_dim: int = 65

    # Контекстные поля (имена колонок в ratings, одна на фичу)
    context_columns: List[str] = field(default_factory=list)
    # Размеры словарей контекстных переменных (в порядке context_columns)
    context_sizes: List[int] = field(default_factory=list)

    # Энкодеры (LabelEncoder'ы) и нормализаторы (MinMaxScaler и т.п.)
    encoders: Dict[str, Any] = field(default_factory=dict)
    scalers: Dict[str, Any] = field(default_factory=dict)

    # Популярность предметов в каталоге: {item_encoded_id: count}
    item_popularity: Dict[int, int] = field(default_factory=dict)

    # OULAD-специфичное: траектории в виде упорядоченных событий
    # [{user_id, step_id, week, criteria: {...}}]
    trajectories: Optional[List[List[Dict[str, Any]]]] = None

    # Маски действий по пользователю (для OULAD), {user_encoded_id: np.ndarray[bool]}
    action_masks: Optional[Dict[int, Any]] = None

    # Прочие метаданные (доля валидации, исходные пути и т.п.)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # --- Удобные хелперы -----------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        """Сводка по bundle (для логирования/ноутбуков)."""
        return {
            "dataset_type": self.dataset_type,
            "n_users": self.n_users,
            "n_items": self.n_items,
            "state_dim": self.state_dim,
            "n_ratings_rows": len(self.ratings),
            "target_columns": list(self.target_columns),
            "context_columns": list(self.context_columns),
            "context_sizes": list(self.context_sizes),
            "has_trajectories": self.trajectories is not None,
            "has_action_masks": self.action_masks is not None,
            "metadata_keys": list(self.metadata.keys()),
        }
