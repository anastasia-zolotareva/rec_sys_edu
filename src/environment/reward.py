"""
Централизованные функции вычисления вознаграждения.

Поддерживаются два режима:

- ``itmrec``: многокритериальная взвешенная награда с контекстно-зависимой
  коррекцией весов, демографическим множителем и бонусом новизны.
- ``oulad``: инкрементное изменение прокси-критериев (Mastery, Engagement,
  SelfRegulation, Outcome) плюс терминальный бонус.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

# ---------------------------------------------------------------------------
# ITM-Rec
# ---------------------------------------------------------------------------

DEFAULT_ITMREC_WEIGHTS = {"w1": 0.50, "w2": 0.30, "w3": 0.15, "w4": 0.05}


def adjust_itmrec_weights(
    context: Mapping[str, int],
    base_weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Корректирует веса критериев под контекст (COVID, класс и т.п.)."""
    base = dict(base_weights or DEFAULT_ITMREC_WEIGHTS)
    weights = dict(base)

    lockdown = int(context.get("lockdown", 0))
    class_id = int(context.get("class", 0))

    if lockdown in (1, 2):
        weights["w3"] = 0.25
        weights["w1"] = max(weights["w1"] - 0.05, 0.0)
    if class_id == 1:  # DB
        weights["w2"] = 0.35

    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    return weights


def demographic_multiplier(demo_vector: np.ndarray) -> float:
    """Корректирует вознаграждение на основе демографических характеристик."""
    multiplier = 1.0
    try:
        married = float(demo_vector[-1])
        age_onehot = demo_vector[1:5]
        if married == 1:
            multiplier *= 0.9
        if np.argmax(age_onehot) in (0, 1):
            multiplier *= 1.1
    except (IndexError, ValueError):
        pass
    return multiplier


def calculate_itmrec_reward(
    feedback: Mapping[str, float],
    context: Mapping[str, int],
    demo_vector: np.ndarray,
    novelty: float,
    novelty_weight: float = 0.05,
    base_weights: Optional[Mapping[str, float]] = None,
) -> float:
    """Вычисляет вознаграждение ITM-Rec шага по мультикритериальному фидбеку.

    ``feedback`` — оценки по шкале 1..5 (App/Data/Ease обязательны).
    
    Итоговая награда ограничена в [0, 1].
    """
    weights = adjust_itmrec_weights(context, base_weights)
    normalized = {
        "app": float(feedback.get("app", 0.0)) / 5.0,
        "data": float(feedback.get("data", 0.0)) / 5.0,
        "ease": float(feedback.get("ease", 0.0)) / 5.0,
    }
    base_reward = (
        weights["w1"] * normalized["app"]
        + weights["w2"] * normalized["data"]
        + weights["w3"] * normalized["ease"]
    )
    reward = base_reward * demographic_multiplier(demo_vector) + novelty_weight * float(novelty)
    # Ограничиваем награду в [0, 1]
    return float(np.clip(reward, 0.0, 1.0))


def calculate_cosine_novelty(
    action: int,
    recommended_items: set,
    item_embeddings_cache: Mapping[int, np.ndarray],
    popularity: Optional[Mapping[int, int]] = None,
    max_popularity: Optional[int] = None,
) -> float:
    """Рассчитывает новизну через косинусную непохожесть на уже рекомендованные предметы.

    Формула: ``novelty = 1 - max(cos_sim(emb_a, emb_j))`` для ``j`` из recommended_items.
    Если история пуста — возвращается ``1 - popularity / max_popularity``.
    """
    if action not in item_embeddings_cache:
        return 1.0

    emb_a = np.asarray(item_embeddings_cache[action], dtype=np.float64)
    norm_a = np.linalg.norm(emb_a)
    if norm_a < 1e-12:
        return 1.0

    max_sim = 0.0
    for other in recommended_items:
        if other == action:
            continue
        emb_b = np.asarray(item_embeddings_cache.get(other, None), dtype=np.float64) if item_embeddings_cache.get(other) is not None else None
        if emb_b is None:
            continue
        norm_b = np.linalg.norm(emb_b)
        if norm_b < 1e-12:
            continue
        sim = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
        max_sim = max(max_sim, sim)

    if not recommended_items:
        if popularity and max_popularity:
            return float(np.clip(1.0 - popularity.get(action, 0) / max(max_popularity, 1), 0.0, 1.0))
        return 1.0

    return float(np.clip(1.0 - max_sim, 0.0, 1.0))


# ---------------------------------------------------------------------------
# OULAD
# ---------------------------------------------------------------------------


DEFAULT_OULAD_WEIGHTS = {
    "outcome": 0.35,
    "mastery": 0.25,
    "engagement": 0.20,
    "selfregulation": 0.20,
}


def calculate_oulad_step_reward(
    prev_proxy: Mapping[str, float],
    new_proxy: Mapping[str, float],
    weights: Optional[Mapping[str, float]] = None,
) -> float:
    """Вычисляет инкрементное вознаграждение как взвешенную сумму приростов прокси-критериев.

    Отрицательные дельты учитываются для повышения информативности обучающей среды.
    """
    w = dict(weights or DEFAULT_OULAD_WEIGHTS)
    delta = 0.0
    for key, weight in w.items():
        delta += weight * (float(new_proxy.get(key, 0.0)) - float(prev_proxy.get(key, 0.0)))
    return float(delta)


def calculate_oulad_terminal_bonus(
    final_proxy: Mapping[str, float],
    is_withdrawn: bool,
    outcome_weight: float = 0.5,
    withdrawn_penalty: float = 0.5,
) -> float:
    """Вычисляет терминальный бонус: поощряет высокие итоговые оценки, штрафует отчисления."""
    outcome = float(final_proxy.get("outcome", 0.0))
    bonus = outcome_weight * outcome
    if is_withdrawn:
        bonus -= withdrawn_penalty
    return float(bonus)


# ---------------------------------------------------------------------------
# Фасад
# ---------------------------------------------------------------------------


def calculate_reward(
    mode: str,
    *,
    feedback: Optional[Mapping[str, float]] = None,
    context: Optional[Mapping[str, int]] = None,
    demo_vector: Optional[np.ndarray] = None,
    novelty: float = 0.0,
    novelty_weight: float = 0.05,
    base_weights: Optional[Mapping[str, float]] = None,
    prev_proxy: Optional[Mapping[str, float]] = None,
    new_proxy: Optional[Mapping[str, float]] = None,
    oulad_weights: Optional[Mapping[str, float]] = None,
) -> float:
    """Единый интерфейс для вычисления вознаграждения в указанном режиме."""
    mode = mode.lower()
    if mode == "itmrec":
        if feedback is None or context is None or demo_vector is None:
            raise ValueError("Для mode='itmrec' нужны feedback, context и demo_vector")
        return calculate_itmrec_reward(
            feedback,
            context,
            np.asarray(demo_vector),
            novelty=novelty,
            novelty_weight=novelty_weight,
            base_weights=base_weights,
        )
    if mode == "oulad":
        if prev_proxy is None or new_proxy is None:
            raise ValueError("Для mode='oulad' нужны prev_proxy и new_proxy")
        return calculate_oulad_step_reward(prev_proxy, new_proxy, oulad_weights)
    raise ValueError(f"Неизвестный reward mode: {mode}")
