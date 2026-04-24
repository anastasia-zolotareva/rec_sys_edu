"""
Построение action-mask для OULAD-среды.

Маска ограничивает выбор RL-агента допустимыми mixed-step действиями:
- шаг относится к текущему code_module/code_presentation;
- шаг не был завершен ранее (для assessment);
- шаг допустим на текущей неделе курса.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence, Set

import numpy as np


def _is_assessment(meta: Mapping[str, Any]) -> bool:
    return str(meta.get("kind", "")).lower() == "assessment"


def build_oulad_action_mask(
    items_meta: Mapping[int, Mapping[str, Any]],
    *,
    current_week: int,
    total_weeks: int,
    completed_items: Iterable[int] = (),
    available_kinds: Optional[Set[str]] = None,
) -> np.ndarray:
    """Строит булеву маску действий для OULAD шага.

    Args:
        items_meta: словарь {item_id: {kind, assessment_type, activity_type, ...}}.
        current_week: текущая неделя курса (0..total_weeks).
        total_weeks: общее число недель курса.
        completed_items: уже завершенные assessment-items (их нельзя повторять).
        available_kinds: разрешенные типы шагов ({"vle", "assessment"}).
            Если None — разрешены все.

    Returns:
        numpy-массив float32 длины n_items со значениями {0.0, 1.0}.
    """
    n_items = len(items_meta)
    mask = np.ones(n_items, dtype=np.float32)
    completed = set(int(x) for x in completed_items)
    restrict_kinds = None
    if available_kinds is not None:
        restrict_kinds = {str(k).lower() for k in available_kinds}

    is_final_phase = current_week + 1 >= max(1, total_weeks)

    for item_id, meta in items_meta.items():
        kind = str(meta.get("kind", "")).lower()
        if restrict_kinds is not None and kind not in restrict_kinds:
            mask[int(item_id)] = 0.0
            continue
        if _is_assessment(meta) and int(item_id) in completed:
            mask[int(item_id)] = 0.0
            continue
        # Assessment экзаменационного типа доступен только в финальную фазу
        if _is_assessment(meta) and str(meta.get("assessment_type", "")).lower() == "exam":
            if not is_final_phase:
                mask[int(item_id)] = 0.0
                continue

    # Гарантируем, что хотя бы одно действие доступно (fallback)
    if mask.sum() == 0:
        mask[:] = 1.0
    return mask


def any_action_available(mask: np.ndarray) -> bool:
    """True, если в маске есть хотя бы одно допустимое действие."""
    return bool(np.any(mask > 0))


def available_action_count(mask: np.ndarray) -> int:
    return int(np.sum(mask > 0))


def action_availability_features(
    mask: np.ndarray,
    items_meta: Mapping[int, Mapping[str, Any]],
) -> np.ndarray:
    """Возвращает вектор признаков (4), описывающих пространство действий:

    - num_available_norm: доля допустимых действий;
    - has_assessment: присутствует ли хотя бы один assessment-action;
    - has_vle: присутствует ли хотя бы один vle-action;
    - has_exam: доступен ли экзамен (финальная фаза).
    """
    total = max(1, len(mask))
    available = 0
    has_assessment = 0.0
    has_vle = 0.0
    has_exam = 0.0

    for idx, m in enumerate(mask):
        if m <= 0:
            continue
        available += 1
        meta = items_meta.get(int(idx)) or {}
        kind = str(meta.get("kind", "")).lower()
        if kind == "assessment":
            has_assessment = 1.0
            if str(meta.get("assessment_type", "")).lower() == "exam":
                has_exam = 1.0
        elif kind == "vle":
            has_vle = 1.0

    return np.array([
        available / total,
        has_assessment,
        has_vle,
        has_exam,
    ], dtype=np.float32)


__all__ = [
    "action_availability_features",
    "any_action_available",
    "available_action_count",
    "build_oulad_action_mask",
]
