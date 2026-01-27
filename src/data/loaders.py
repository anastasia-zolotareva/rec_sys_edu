"""
Модуль загрузки данных с Kaggle.
"""

import os
from pathlib import Path
from typing import Optional
import pandas as pd
import kagglehub


def download_kaggle_dataset(
    dataset_name: str = "irecsys/itmrec",
    output_dir: str = "data/raw"
) -> str:
    """
    Загрузка датасета с Kaggle.
    
    Args:
        dataset_name: Имя датасета в формате "username/dataset-name"
        output_dir: Директория для сохранения данных
    
    Returns:
        Путь к загруженным данным
    
    Raises:
        Exception: Если загрузка не удалась
    """
    print(f"Загрузка датасета {dataset_name} с Kaggle...")
    
    try:
        # Создание директории если не существует
        os.makedirs(output_dir, exist_ok=True)
        
        # Загрузка датасета
        path = kagglehub.dataset_download(dataset_name)
        
        print(f"Датасет загружен в: {path}")
        return path
    
    except Exception as e:
        print(f"Ошибка при загрузке датасета: {e}")
        raise


def load_ratings(path: str) -> pd.DataFrame:
    """
    Загрузка файла ratings.csv.
    
    Args:
        path: Путь к директории с данными или к файлу ratings.csv
    
    Returns:
        DataFrame с рейтингами
    """
    if os.path.isfile(path):
        ratings_path = path
    else:
        ratings_path = os.path.join(path, "ratings.csv")
    
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"Файл не найден: {ratings_path}")
    
    ratings = pd.read_csv(ratings_path)
    print(f"Загружено {len(ratings)} записей рейтингов")
    return ratings


def load_users(path: str) -> pd.DataFrame:
    """
    Загрузка файла users.csv.
    
    Args:
        path: Путь к директории с данными или к файлу users.csv
    
    Returns:
        DataFrame с пользователями
    """
    if os.path.isfile(path):
        users_path = path
    else:
        users_path = os.path.join(path, "users.csv")
    
    if not os.path.exists(users_path):
        raise FileNotFoundError(f"Файл не найден: {users_path}")
    
    users = pd.read_csv(users_path)
    print(f"Загружено {len(users)} пользователей")
    return users


def load_items(path: str) -> pd.DataFrame:
    """
    Загрузка файла items.csv.
    
    Args:
        path: Путь к директории с данными или к файлу items.csv
    
    Returns:
        DataFrame с предметами
    """
    if os.path.isfile(path):
        items_path = path
    else:
        items_path = os.path.join(path, "items.csv")
    
    if not os.path.exists(items_path):
        raise FileNotFoundError(f"Файл не найден: {items_path}")
    
    items = pd.read_csv(items_path)
    print(f"Загружено {len(items)} предметов")
    return items


def load_group_ratings(path: str) -> pd.DataFrame:
    """
    Загрузка файла group_ratings.csv (опционально).
    
    Args:
        path: Путь к директории с данными или к файлу group_ratings.csv
    
    Returns:
        DataFrame с групповыми рейтингами или None если файл не найден
    """
    if os.path.isfile(path):
        group_ratings_path = path
    else:
        group_ratings_path = os.path.join(path, "group_ratings.csv")
    
    if not os.path.exists(group_ratings_path):
        print(f"Файл group_ratings.csv не найден, пропускаем")
        return None
    
    group_ratings = pd.read_csv(group_ratings_path)
    print(f"Загружено {len(group_ratings)} групповых рейтингов")
    return group_ratings


def load_all_data(
    data_path: str,
    load_group_ratings: bool = False
) -> dict:
    """
    Загрузка всех данных датасета.
    
    Args:
        data_path: Путь к директории с данными
        load_group_ratings: Загружать ли group_ratings.csv
    
    Returns:
        Словарь с загруженными данными:
        {
            'ratings': DataFrame,
            'users': DataFrame,
            'items': DataFrame,
            'group_ratings': DataFrame или None
        }
    """
    print("=" * 50)
    print("ЗАГРУЗКА ДАТАСЕТА ITM-REC")
    print("=" * 50)
    
    data = {
        'ratings': load_ratings(data_path),
        'users': load_users(data_path),
        'items': load_items(data_path),
    }
    
    if load_group_ratings:
        data['group_ratings'] = load_group_ratings(data_path)
    
    print(f"\nРазмеры загруженных таблиц:")
    print(f"- ratings: {data['ratings'].shape}")
    print(f"- users: {data['users'].shape}")
    print(f"- items: {data['items'].shape}")
    if 'group_ratings' in data and data['group_ratings'] is not None:
        print(f"- group_ratings: {data['group_ratings'].shape}")
    
    return data
