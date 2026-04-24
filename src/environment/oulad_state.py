"""
Кодирование состояния для OULAD-среды.

Формирует вектор состояния размерности ``state_dim`` (по умолчанию 96), включающий:
- embedding пользователя (32);
- one-hot курса/презентации (модули + сессии);
- прогресс и дедлайны (текущая неделя, оставшиеся недели, ratio сданных assessments, ...);
- демография / академический фон (gender, age_band, imd, disability, credits, prev_attempts);
- агрегированная история mixed-step (средние proxy, std, последнее значение);
- временные скаляры;
- признаки доступности действий;
- текущие proxy-критерии (Mastery, Engagement, SelfRegulation, Outcome).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


DEFAULT_STATE_DIM = 96

HIGHEST_EDUCATION_ORDER = [
    "no formal quals",
    "lower than a level",
    "a level or equivalent",
    "he qualification",
    "post graduate qualification",
]

IMD_ORDER = [
    "0-10%",
    "10-20",
    "20-30%",
    "30-40%",
    "40-50%",
    "50-60%",
    "60-70%",
    "70-80%",
    "80-90%",
    "90-100%",
]

AGE_ORDER = ["0-35", "35-55", "55<="]


def _onehot(index: int, size: int) -> np.ndarray:
    vec = np.zeros(size, dtype=np.float32)
    if 0 <= index < size:
        vec[index] = 1.0
    return vec


def _safe_norm(value: float, maximum: float) -> float:
    if maximum <= 0:
        return 0.0
    return float(np.clip(value / maximum, 0.0, 1.0))


def _lookup_in_order(value: Any, order: Sequence[str]) -> int:
    if value is None:
        return -1
    text = str(value).strip().lower()
    for idx, key in enumerate(order):
        if text.startswith(key.split()[0]):
            return idx
    return -1


def build_oulad_demographics(
    user_row: Optional[Mapping[str, Any]],
    n_prev_attempts_cap: int = 6,
    credits_cap: float = 600.0,
) -> np.ndarray:
    """12-мерный вектор академического/демографического фона."""
    if user_row is None:
        return np.zeros(12, dtype=np.float32)

    gender = 1.0 if str(user_row.get("gender", "")).lower().startswith("m") else 0.0
    age_idx = _lookup_in_order(user_row.get("age_band"), AGE_ORDER)
    imd_idx = _lookup_in_order(user_row.get("imd_band"), IMD_ORDER)
    disability = 1.0 if str(user_row.get("disability", "")).lower() == "y" else 0.0

    num_attempts = _safe_norm(float(user_row.get("num_of_prev_attempts", 0) or 0), float(n_prev_attempts_cap))
    studied_credits = _safe_norm(float(user_row.get("studied_credits", 0) or 0), float(credits_cap))

    age_onehot = _onehot(age_idx if age_idx >= 0 else 0, len(AGE_ORDER))
    imd_norm = 0.0 if imd_idx < 0 else imd_idx / max(1, len(IMD_ORDER) - 1)
    he_idx = _lookup_in_order(user_row.get("highest_education"), HIGHEST_EDUCATION_ORDER)
    he_norm = 0.0 if he_idx < 0 else he_idx / max(1, len(HIGHEST_EDUCATION_ORDER) - 1)

    return np.array([
        gender,
        *age_onehot.tolist(),
        imd_norm,
        disability,
        num_attempts,
        studied_credits,
        he_norm,
        1.0 if age_idx >= 1 else 0.0,  # флаг "возраст >= 35"
        1.0 if num_attempts > 0 else 0.0,
    ], dtype=np.float32)[:12]


def aggregate_history(
    trajectory: Sequence[Mapping[str, Any]],
    last_k: int = 3,
) -> np.ndarray:
    """16-мерный агрегат mixed-step истории: средние/стд за последние ``last_k`` шагов.

    Структура:
    - mean(M, E, SR, O) (4)
    - std(M, E, SR, O) (4)
    - last(M, E, SR, O) (4)
    - overall mean(M, E, SR, O) (4)
    """
    zeros = np.zeros(16, dtype=np.float32)
    if not trajectory:
        return zeros
    fields = ("mastery", "engagement", "selfregulation", "outcome")
    last_window = list(trajectory[-last_k:])

    def _extract(vals: Sequence[Mapping[str, Any]], field_name: str) -> np.ndarray:
        return np.array([float(v.get(field_name, 0.0)) for v in vals], dtype=np.float32)

    last_mean = np.array([_extract(last_window, f).mean() for f in fields], dtype=np.float32)
    last_std = np.array([_extract(last_window, f).std() for f in fields], dtype=np.float32)
    last_value = np.array([_extract(last_window, f)[-1] for f in fields], dtype=np.float32)
    overall_mean = np.array([_extract(trajectory, f).mean() for f in fields], dtype=np.float32)
    return np.concatenate([last_mean, last_std, last_value, overall_mean]).astype(np.float32)


def _pad_or_trim(vec: np.ndarray, size: int) -> np.ndarray:
    if vec.shape[0] == size:
        return vec
    if vec.shape[0] > size:
        return vec[:size]
    out = np.zeros(size, dtype=np.float32)
    out[: vec.shape[0]] = vec
    return out


def encode_oulad_state(
    *,
    user_embedding: np.ndarray,
    module_index: int,
    presentation_index: int,
    n_modules: int,
    n_presentations: int,
    user_row: Optional[Mapping[str, Any]],
    current_week: int,
    total_weeks: int,
    trajectory: Sequence[Mapping[str, Any]],
    current_proxy: Mapping[str, float],
    availability: np.ndarray,
    max_trajectory_length: int,
    state_dim: int = DEFAULT_STATE_DIM,
) -> np.ndarray:
    """Собирает финальный state-вектор.

    Аргументы:
        user_embedding: эмбеддинг студента (32-dim).
        module_index/presentation_index: индексы для one-hot.
        n_modules/n_presentations: размер словарей.
        user_row: строка из ``DatasetBundle.users`` (или ``None``).
        current_week/total_weeks: прогресс курса.
        trajectory: список словарей с ключами mastery/engagement/selfregulation/outcome.
        current_proxy: текущее состояние прокси-критериев (dict).
        availability: вектор признаков action-mask (4).
        max_trajectory_length: максимум для нормализации счётчика шагов.
        state_dim: требуемая размерность (по умолчанию 96).
    """
    user_embedding = np.asarray(user_embedding, dtype=np.float32).flatten()
    components: List[np.ndarray] = []

    components.append(_pad_or_trim(user_embedding, 32))

    module_slot = max(n_modules, 8)
    presentation_slot = max(n_presentations, 4)
    components.append(_pad_or_trim(_onehot(module_index, module_slot), module_slot))
    components.append(_pad_or_trim(_onehot(presentation_index, presentation_slot), presentation_slot))

    progress = float(current_week) / max(1, total_weeks)
    remaining = 1.0 - progress
    step_count_norm = len(trajectory) / max(1, max_trajectory_length)

    progress_vec = np.array([
        progress,
        remaining,
        step_count_norm,
        1.0 if progress >= 0.75 else 0.0,
        1.0 if progress >= 0.9 else 0.0,
        float(current_proxy.get("mastery", 0.0)),
        float(current_proxy.get("engagement", 0.0)),
        float(current_proxy.get("selfregulation", 0.0)),
        float(current_proxy.get("outcome", 0.0)),
        float(np.clip(total_weeks / 52.0, 0.0, 1.0)),
    ], dtype=np.float32)
    components.append(progress_vec)

    components.append(build_oulad_demographics(user_row))
    components.append(aggregate_history(trajectory))

    time_scalars = np.array([
        progress,
        step_count_norm,
        float(len(trajectory)) / 10.0,
        float(np.mean([t.get("reward", 0.0) for t in trajectory])) if trajectory else 0.0,
        float(np.max([t.get("reward", 0.0) for t in trajectory])) if trajectory else 0.0,
        float(np.min([t.get("reward", 0.0) for t in trajectory])) if trajectory else 0.0,
    ], dtype=np.float32)
    components.append(time_scalars)

    components.append(_pad_or_trim(np.asarray(availability, dtype=np.float32), 4))

    current_proxy_vec = np.array([
        float(current_proxy.get("mastery", 0.0)),
        float(current_proxy.get("engagement", 0.0)),
        float(current_proxy.get("selfregulation", 0.0)),
        float(current_proxy.get("outcome", 0.0)),
    ], dtype=np.float32)
    components.append(current_proxy_vec)

    vec = np.concatenate(components).astype(np.float32)
    return _pad_or_trim(vec, state_dim)


__all__ = [
    "DEFAULT_STATE_DIM",
    "aggregate_history",
    "build_oulad_demographics",
    "encode_oulad_state",
]
