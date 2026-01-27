"""
Функции кодирования состояний для EducationalEnvironment.
"""

import numpy as np
from typing import Dict, List, Optional


def encode_context(context: Dict[str, int], trajectory: Optional[List] = None) -> np.ndarray:
    """
    Кодирование контекстных переменных.
    
    Args:
        context: Словарь с контекстными переменными:
            - 'class': ID класса (0-2)
            - 'semester': ID семестра (0-1)
            - 'lockdown': ID периода COVID (0-2)
        trajectory: История взаимодействий для расчета success_rate
    
    Returns:
        Вектор контекста размерности 10:
        - class_onehot: 3 измерения
        - semester_onehot: 2 измерения
        - lockdown_onehot: 3 измерения
        - time_in_semester: 1 измерение
        - success_rate: 1 измерение
    """
    # One-hot кодирование
    class_onehot = np.zeros(3)  # DA, DB, DM
    class_onehot[context['class']] = 1
    
    semester_onehot = np.zeros(2)  # Fall, Spring
    semester_onehot[context['semester']] = 1
    
    lockdown_onehot = np.zeros(3)  # PRE, DUR, POS
    lockdown_onehot[context['lockdown']] = 1
    
    # Время в семестре (нормализованное)
    time_in_semester = np.random.random()  # Для симуляции
    
    # Успешность предыдущих рекомендаций
    if trajectory is not None and len(trajectory) > 0:
        success_rate = np.mean([1 if r.get('reward', 0) > 0.5 else 0 for r in trajectory])
    else:
        success_rate = 0.5
    
    context_vector = np.concatenate([
        class_onehot,
        semester_onehot,
        lockdown_onehot,
        [time_in_semester],
        [success_rate]
    ])
    
    return context_vector


def encode_history(
    trajectory: List[Dict],
    item_embeddings_cache: Dict[int, np.ndarray],
    max_interactions: int = 3
) -> np.ndarray:
    """
    Кодирование истории взаимодействий.
    
    Args:
        trajectory: Список взаимодействий, каждое содержит:
            - 'item_id': ID предмета
            - 'rating': Рейтинг
            - 'app', 'data', 'ease': Критерии оценки
            - 'time_since': Время с момента взаимодействия
        item_embeddings_cache: Кэш эмбеддингов предметов
        max_interactions: Максимальное количество взаимодействий в истории
    
    Returns:
        Вектор истории размерности 15 (3 взаимодействия × 5 признаков):
        - item_emb_compressed: 4 измерения (первые 4 из эмбеддинга)
        - rating_norm: 1 измерение
        - time_since: 1 измерение
        - app, data, ease: 3 измерения
    """
    if len(trajectory) == 0:
        return np.zeros(15)  # 3 interactions × 5 features
    
    # Берем последние N взаимодействий
    recent_history = trajectory[-max_interactions:]
    
    history_vectors = []
    for interaction in recent_history:
        item_id = interaction['item_id']
        
        # Эмбеддинг предмета (первые 4 измерения)
        if item_id in item_embeddings_cache:
            item_emb = item_embeddings_cache[item_id]
            item_emb_compressed = item_emb[:4]  # Сжатие до 4 измерений
        else:
            item_emb_compressed = np.zeros(4)
        
        # Нормализованный рейтинг
        rating_norm = interaction.get('rating', 0) / 5.0
        
        # Время с момента взаимодействия (нормализованное)
        time_since = interaction.get('time_since', 0.1)
        
        # Дополнительные признаки
        additional_features = np.array([
            interaction.get('app', 0.5) / 5.0,
            interaction.get('data', 0.5) / 5.0,
            interaction.get('ease', 0.5) / 5.0
        ])
        
        interaction_vector = np.concatenate([
            item_emb_compressed,
            [rating_norm, time_since],
            additional_features
        ])
        history_vectors.append(interaction_vector)
    
    # Дополнение нулями, если истории недостаточно
    while len(history_vectors) < max_interactions:
        history_vectors.append(np.zeros(5))
    
    # Объединение
    full_history = np.concatenate(history_vectors)
    
    # Проверка размерности
    expected_dim = max_interactions * 5
    if len(full_history) != expected_dim:
        # Обрезка или дополнение
        full_history = np.pad(full_history, (0, max(0, expected_dim - len(full_history))))[:expected_dim]
    
    return full_history


def get_demographic_vector(
    user_id: int,
    users_df,
    user_encoder
) -> np.ndarray:
    """
    Получение демографического вектора пользователя.
    
    Args:
        user_id: Закодированный ID пользователя
        users_df: DataFrame с пользователями
        user_encoder: LabelEncoder для декодирования user_id
    
    Returns:
        Вектор демографических признаков размерности 6:
        - gender: 1 измерение (0=Female, 1=Male)
        - age_onehot: 4 измерения (<20, 20-25, 26-30, >30)
        - married: 1 измерение (0=Single, 1=Married)
    """
    try:
        # Поиск реального UserID
        original_user_id = user_encoder.inverse_transform([user_id])[0]
        user_data = users_df[users_df['UserID'] == original_user_id]
        
        if len(user_data) == 0:
            # Данные по умолчанию
            return np.array([0, 0, 0, 0, 0, 1])  # Female, 26-30, не женат
        
        row = user_data.iloc[0]
        
        # Gender (0=Female, 1=Male)
        # Обработка возможных вариантов названия колонки
        gender_col = None
        for col in users_df.columns:
            if 'gender' in col.lower() or col.strip() == ' Gender':
                gender_col = col
                break
        
        if gender_col is not None:
            gender = 1 if row[gender_col] == 1 else 0
        else:
            gender = 0
        
        # Age group one-hot
        age_col = None
        for col in users_df.columns:
            if 'age' in col.lower() or col.strip() == ' Age':
                age_col = col
                break
        
        if age_col is not None:
            age = row[age_col]
            age_onehot = np.zeros(4)
            if age == '<20':
                age_onehot[0] = 1
            elif age == '20-25':
                age_onehot[1] = 1
            elif age == '26-30':
                age_onehot[2] = 1
            else:
                age_onehot[3] = 1
        else:
            age_onehot = np.array([0, 0, 1, 0])  # По умолчанию 26-30
        
        # Married (0=Single, 1=Married)
        married_col = None
        for col in users_df.columns:
            if 'married' in col.lower():
                married_col = col
                break
        
        if married_col is not None:
            married = 1 if row[married_col] == 1 else 0
        else:
            married = 0
        
        return np.concatenate([[gender], age_onehot, [married]])
    
    except Exception as e:
        # В случае ошибки возвращаем значения по умолчанию
        print(f"Предупреждение: не удалось получить демографические данные для user_id={user_id}: {e}")
        return np.array([0, 0, 0, 0, 0, 1])  # Female, 26-30, не женат
