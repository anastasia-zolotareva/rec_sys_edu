"""
Модуль предобработки данных.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


def fill_missing_values(
    df: pd.DataFrame,
    strategy: str = "median",
    columns: Optional[list] = None
) -> pd.DataFrame:
    """
    Заполнение пропущенных значений.
    
    Args:
        df: DataFrame для обработки
        strategy: Стратегия заполнения ('median', 'mean', 'mode', 'zero')
        columns: Список колонок для обработки (None = все числовые)
    
    Returns:
        DataFrame с заполненными пропусками
    """
    df = df.copy()
    
    if columns is None:
        # Выбираем только числовые колонки
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if df[col].isna().any():
            if strategy == "median":
                fill_value = df[col].median()
            elif strategy == "mean":
                fill_value = df[col].mean()
            elif strategy == "mode":
                fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
            elif strategy == "zero":
                fill_value = 0
            else:
                raise ValueError(f"Неизвестная стратегия: {strategy}")
            
            df[col].fillna(fill_value, inplace=True)
            print(f"Заполнено {df[col].isna().sum()} пропусков в колонке {col} значением {fill_value:.2f}")
    
    return df


def encode_categorical(
    df: pd.DataFrame,
    columns: list,
    encoders: Optional[Dict[str, Any]] = None
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Кодирование категориальных признаков.
    
    Args:
        df: DataFrame для обработки
        columns: Список категориальных колонок для кодирования
        encoders: Словарь существующих энкодеров (для тестовых данных)
    
    Returns:
        Tuple (обработанный DataFrame, словарь энкодеров)
    """
    from sklearn.preprocessing import LabelEncoder
    
    df = df.copy()
    
    if encoders is None:
        encoders = {}
        for col in columns:
            if col in df.columns:
                encoders[col] = LabelEncoder()
                df[f"{col}_encoded"] = encoders[col].fit_transform(df[col])
            else:
                print(f"Предупреждение: колонка {col} не найдена")
    else:
        # Используем существующие энкодеры (для тестовых данных)
        for col in columns:
            if col in df.columns and col in encoders:
                df[f"{col}_encoded"] = encoders[col].transform(df[col])
    
    return df, encoders


def normalize_ratings(
    df: pd.DataFrame,
    rating_columns: list = ['Rating', 'App', 'Data', 'Ease'],
    method: str = "minmax"
) -> pd.DataFrame:
    """
    Нормализация рейтингов.
    
    Args:
        df: DataFrame с рейтингами
        rating_columns: Список колонок с рейтингами
        method: Метод нормализации ('minmax', 'standard', 'divide_by_max')
    
    Returns:
        DataFrame с нормализованными рейтингами
    """
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    
    df = df.copy()
    
    for col in rating_columns:
        if col not in df.columns:
            continue
        
        if method == "minmax":
            scaler = MinMaxScaler()
            df[f"{col}_norm"] = scaler.fit_transform(df[[col]])
        elif method == "standard":
            scaler = StandardScaler()
            df[f"{col}_norm"] = scaler.fit_transform(df[[col]])
        elif method == "divide_by_max":
            max_val = df[col].max()
            df[f"{col}_norm"] = df[col] / max_val if max_val > 0 else df[col]
        else:
            raise ValueError(f"Неизвестный метод нормализации: {method}")
    
    return df


def validate_data(
    ratings: pd.DataFrame,
    users: pd.DataFrame,
    items: pd.DataFrame
) -> bool:
    """
    Валидация загруженных данных.
    
    Args:
        ratings: DataFrame с рейтингами
        users: DataFrame с пользователями
        items: DataFrame с предметами
    
    Returns:
        True если данные валидны
    
    Raises:
        ValueError: Если данные невалидны
    """
    # Проверка обязательных колонок
    required_ratings_cols = ['UserID', 'Item', 'Rating']
    required_users_cols = ['UserID']
    required_items_cols = ['Item']
    
    for col in required_ratings_cols:
        if col not in ratings.columns:
            raise ValueError(f"Отсутствует обязательная колонка в ratings: {col}")
    
    for col in required_users_cols:
        if col not in users.columns:
            raise ValueError(f"Отсутствует обязательная колонка в users: {col}")
    
    for col in required_items_cols:
        if col not in items.columns:
            raise ValueError(f"Отсутствует обязательная колонка в items: {col}")
    
    # Проверка соответствия ID
    rating_user_ids = set(ratings['UserID'].unique())
    user_ids = set(users['UserID'].unique())
    
    if not rating_user_ids.issubset(user_ids):
        missing = rating_user_ids - user_ids
        print(f"Предупреждение: {len(missing)} пользователей из ratings отсутствуют в users")
    
    rating_item_ids = set(ratings['Item'].unique())
    item_ids = set(items['Item'].unique())
    
    if not rating_item_ids.issubset(item_ids):
        missing = rating_item_ids - item_ids
        print(f"Предупреждение: {len(missing)} предметов из ratings отсутствуют в items")
    
    print("Валидация данных пройдена успешно")
    return True
