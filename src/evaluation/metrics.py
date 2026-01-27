"""
Функции расчета метрик качества рекомендаций.
"""

import numpy as np
from typing import List, Optional
from sklearn.metrics import precision_score, recall_score, f1_score


def calculate_precision_at_k(
    recommendations: List[int],
    ground_truth: List[int],
    k: int,
    n_items: Optional[int] = None
) -> float:
    """
    Расчет Precision@K.
    
    Args:
        recommendations: Список рекомендованных предметов
        ground_truth: Список релевантных предметов
        k: Количество рекомендаций для оценки
        n_items: Общее количество предметов (для бинарного кодирования)
    
    Returns:
        Precision@K (0-1)
    """
    if n_items is None:
        n_items = max(max(recommendations, default=0), max(ground_truth, default=0)) + 1
    
    y_true = np.zeros(n_items)
    y_pred = np.zeros(n_items)
    
    y_true[list(ground_truth)] = 1
    y_pred[recommendations[:k]] = 1
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    return float(precision)


def calculate_recall_at_k(
    recommendations: List[int],
    ground_truth: List[int],
    k: int,
    n_items: Optional[int] = None
) -> float:
    """
    Расчет Recall@K.
    
    Args:
        recommendations: Список рекомендованных предметов
        ground_truth: Список релевантных предметов
        k: Количество рекомендаций для оценки
        n_items: Общее количество предметов (для бинарного кодирования)
    
    Returns:
        Recall@K (0-1)
    """
    if n_items is None:
        n_items = max(max(recommendations, default=0), max(ground_truth, default=0)) + 1
    
    y_true = np.zeros(n_items)
    y_pred = np.zeros(n_items)
    
    y_true[list(ground_truth)] = 1
    y_pred[recommendations[:k]] = 1
    
    recall = recall_score(y_true, y_pred, zero_division=0)
    return float(recall)


def calculate_f1_at_k(
    recommendations: List[int],
    ground_truth: List[int],
    k: int,
    n_items: Optional[int] = None
) -> float:
    """
    Расчет F1@K.
    
    Args:
        recommendations: Список рекомендованных предметов
        ground_truth: Список релевантных предметов
        k: Количество рекомендаций для оценки
        n_items: Общее количество предметов (для бинарного кодирования)
    
    Returns:
        F1@K (0-1)
    """
    precision = calculate_precision_at_k(recommendations, ground_truth, k, n_items)
    recall = calculate_recall_at_k(recommendations, ground_truth, k, n_items)
    
    if (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    return float(f1)


def calculate_coverage(
    recommendations: List[int],
    n_items: int
) -> float:
    """
    Расчет Coverage (покрытие каталога).
    
    Coverage = количество уникальных рекомендованных предметов / общее количество предметов
    
    Args:
        recommendations: Список рекомендованных предметов
        n_items: Общее количество предметов в каталоге
    
    Returns:
        Coverage (0-1)
    """
    unique_recommendations = len(set(recommendations))
    coverage = unique_recommendations / n_items if n_items > 0 else 0.0
    return float(coverage)


def calculate_diversity(
    recommendations: List[int],
    item_embeddings: Optional[np.ndarray] = None
) -> float:
    """
    Расчет Diversity (разнообразие рекомендаций).
    
    Использует простое измерение: доля уникальных пар предметов.
    Если предоставлены эмбеддинги, можно использовать косинусное расстояние.
    
    Args:
        recommendations: Список рекомендованных предметов
        item_embeddings: Опциональные эмбеддинги предметов для более точного расчета
    
    Returns:
        Diversity (0-1), где 1 - максимальное разнообразие
    """
    if len(recommendations) < 2:
        return 0.0
    
    if item_embeddings is not None:
        # Используем косинусное расстояние между эмбеддингами
        from sklearn.metrics.pairwise import cosine_similarity
        
        unique_items = list(set(recommendations))
        if len(unique_items) < 2:
            return 0.0
        
        embeddings = item_embeddings[unique_items]
        similarities = cosine_similarity(embeddings)
        
        # Diversity = 1 - средняя схожесть
        # Исключаем диагональ (схожесть предмета с самим собой = 1)
        mask = np.eye(len(similarities), dtype=bool)
        similarities[mask] = 0
        
        avg_similarity = similarities.sum() / (len(similarities) * (len(similarities) - 1))
        diversity = 1.0 - avg_similarity
    else:
        # Простая метрика: доля уникальных пар
        diversity = 0
        count = 0
        for i in range(len(recommendations)):
            for j in range(i + 1, len(recommendations)):
                diversity += 1 if recommendations[i] != recommendations[j] else 0
                count += 1
        diversity = diversity / count if count > 0 else 0.0
    
    return float(np.clip(diversity, 0.0, 1.0))


def calculate_novelty(
    recommendations: List[int],
    item_popularity: dict,
    k: Optional[int] = None
) -> float:
    """
    Расчет Novelty (новизна рекомендаций).
    
    Novelty = средняя обратная популярность рекомендованных предметов.
    
    Args:
        recommendations: Список рекомендованных предметов
        item_popularity: Словарь {item_id: popularity_count}
        k: Количество рекомендаций для оценки (если None, используется все)
    
    Returns:
        Novelty (0-1), где 1 - максимальная новизна
    """
    if k is not None:
        recommendations = recommendations[:k]
    
    if not recommendations:
        return 0.0
    
    # Нормализация популярности
    if item_popularity:
        max_pop = max(item_popularity.values())
        if max_pop > 0:
            novelty_scores = [
                1.0 - (item_popularity.get(item, 0) / max_pop)
                for item in recommendations
            ]
        else:
            novelty_scores = [1.0] * len(recommendations)
    else:
        novelty_scores = [1.0] * len(recommendations)
    
    return float(np.mean(novelty_scores))


def calculate_cumulative_discounted_reward(
    rewards: List[float],
    gamma: float = 0.99
) -> float:
    """
    Расчет Cumulative Discounted Reward (CDR).
    
    CDR = sum(r_t * γ^t) для всех шагов t
    
    Args:
        rewards: Список наград за шаги эпизода
        gamma: Коэффициент дисконтирования
    
    Returns:
        Cumulative Discounted Reward
    """
    cdr = 0.0
    discount = 1.0
    
    for reward in rewards:
        cdr += reward * discount
        discount *= gamma
    
    return float(cdr)


def calculate_retention_rate(
    rewards: List[float],
    threshold: float = 0.5
) -> float:
    """
    Расчет Retention Rate (удержание пользователей).
    
    Retention Rate = доля шагов с наградой выше порога
    
    Args:
        rewards: Список наград за шаги эпизода
        threshold: Порог для определения "удержания"
    
    Returns:
        Retention Rate (0-1)
    """
    if not rewards:
        return 0.0
    
    retained_steps = sum(1 for r in rewards if r > threshold)
    retention_rate = retained_steps / len(rewards)
    
    return float(retention_rate)
