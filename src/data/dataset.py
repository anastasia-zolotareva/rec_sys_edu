"""
PyTorch Dataset для ITM-REC датасета.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from typing import Tuple, Optional


class ITMDataset(Dataset):
    """
    Датасет для обучения DeepFM-SVD++.
    
    Выполняет:
    - Кодирование категориальных признаков
    - Нормализацию рейтингов
    - Подготовку данных для обучения модели
    
    Attributes:
        ratings: DataFrame с рейтингами (с закодированными колонками)
        users: DataFrame с пользователями
        items: DataFrame с предметами
        n_users: Количество уникальных пользователей
        n_items: Количество уникальных предметов
        n_classes: Количество классов (специализаций)
        n_semesters: Количество семестров
        n_lockdowns: Количество периодов COVID
        user_encoder: LabelEncoder для пользователей
        item_encoder: LabelEncoder для предметов
        class_encoder: LabelEncoder для классов
        semester_encoder: LabelEncoder для семестров
        lockdown_encoder: LabelEncoder для периодов COVID
        rating_scaler: MinMaxScaler для нормализации рейтингов
    """
    
    def __init__(
        self,
        ratings_df: pd.DataFrame,
        users_df: pd.DataFrame,
        items_df: pd.DataFrame
    ):
        """
        Инициализация датасета.
        
        Args:
            ratings_df: DataFrame с рейтингами
            users_df: DataFrame с пользователями
            items_df: DataFrame с предметами
        """
        # Кодирование категориальных признаков
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.class_encoder = LabelEncoder()
        self.semester_encoder = LabelEncoder()
        self.lockdown_encoder = LabelEncoder()
        
        # Подготовка данных
        self.ratings = ratings_df.copy()
        self.users = users_df.copy()
        self.items = items_df.copy()
        
        # Заполнение пропусков в Data колонке (если есть)
        if 'Data' in self.ratings.columns:
            self.ratings['Data'].fillna(self.ratings['Data'].median(), inplace=True)
        
        # Кодирование
        self.ratings['UserID_encoded'] = self.user_encoder.fit_transform(self.ratings['UserID'])
        self.ratings['ItemID_encoded'] = self.item_encoder.fit_transform(self.ratings['Item'])
        self.ratings['Class_encoded'] = self.class_encoder.fit_transform(self.ratings['Class'])
        self.ratings['Semester_encoded'] = self.semester_encoder.fit_transform(self.ratings['Semester'])
        self.ratings['Lockdown_encoded'] = self.lockdown_encoder.fit_transform(self.ratings['Lockdown'])
        
        # Нормализация рейтингов
        self.rating_scaler = MinMaxScaler()
        self.ratings['Rating_norm'] = self.rating_scaler.fit_transform(
            self.ratings[['Rating']]
        ).flatten()
        
        # Размерности
        self.n_users = len(self.user_encoder.classes_)
        self.n_items = len(self.item_encoder.classes_)
        self.n_classes = len(self.class_encoder.classes_)
        self.n_semesters = len(self.semester_encoder.classes_)
        self.n_lockdowns = len(self.lockdown_encoder.classes_)
        
        print(f"Датасет инициализирован:")
        print(f"  Пользователей: {self.n_users}")
        print(f"  Предметов: {self.n_items}")
        print(f"  Классов: {self.n_classes}")
        print(f"  Семестров: {self.n_semesters}")
        print(f"  Периодов COVID: {self.n_lockdowns}")
        print(f"  Записей рейтингов: {len(self.ratings)}")
    
    def __len__(self) -> int:
        """Возвращает размер датасета."""
        return len(self.ratings)
    
    def __getitem__(self, idx: int) -> Tuple[dict, torch.Tensor]:
        """
        Получение одного примера из датасета.
        
        Args:
            idx: Индекс примера
        
        Returns:
            Tuple (features, targets):
            - features: Словарь с признаками
            - targets: Tensor с целевыми значениями (Rating, App, Data, Ease)
        """
        row = self.ratings.iloc[idx]
        
        # Мультикритериальные таргеты
        targets = torch.FloatTensor([
            row['Rating_norm'],
            row['App'] / 5.0,  # Нормализация к [0, 1]
            row['Data'] / 5.0,
            row['Ease'] / 5.0
        ])
        
        # Признаки
        features = {
            'user_id': torch.LongTensor([row['UserID_encoded']]),
            'item_id': torch.LongTensor([row['ItemID_encoded']]),
            'class': torch.LongTensor([row['Class_encoded']]),
            'semester': torch.LongTensor([row['Semester_encoded']]),
            'lockdown': torch.LongTensor([row['Lockdown_encoded']]),
        }
        
        return features, targets
    
    def train_test_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        batch_size: int = 256,
        shuffle: bool = True,
        random_state: Optional[int] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Разделение датасета на train/val/test и создание DataLoader'ов.
        
        Args:
            train_ratio: Доля обучающей выборки
            val_ratio: Доля валидационной выборки
            test_ratio: Доля тестовой выборки
            batch_size: Размер батча
            shuffle: Перемешивать ли данные
            random_state: Seed для воспроизводимости
        
        Returns:
            Tuple (train_loader, val_loader, test_loader)
        
        Raises:
            ValueError: Если сумма долей не равна 1.0
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Сумма train_ratio, val_ratio и test_ratio должна быть равна 1.0")
        
        # Перемешивание индексов
        indices = np.arange(len(self.ratings))
        if shuffle:
            if random_state is not None:
                np.random.seed(random_state)
            np.random.shuffle(indices)
        
        # Разделение
        n_train = int(len(indices) * train_ratio)
        n_val = int(len(indices) * val_ratio)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        # Создание подмножеств
        train_subset = torch.utils.data.Subset(self, train_indices)
        val_subset = torch.utils.data.Subset(self, val_indices)
        test_subset = torch.utils.data.Subset(self, test_indices)
        
        # Создание DataLoader'ов
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        print(f"Разделение датасета:")
        print(f"  Train: {len(train_indices)} ({len(train_indices)/len(indices)*100:.1f}%)")
        print(f"  Val: {len(val_indices)} ({len(val_indices)/len(indices)*100:.1f}%)")
        print(f"  Test: {len(test_indices)} ({len(test_indices)/len(indices)*100:.1f}%)")
        
        return train_loader, val_loader, test_loader
    
    def get_user_history(self, user_id: int, max_items: int = 10) -> list:
        """
        Получение истории взаимодействий пользователя.
        
        Args:
            user_id: Закодированный ID пользователя
            max_items: Максимальное количество предметов в истории
        
        Returns:
            Список закодированных ID предметов
        """
        user_ratings = self.ratings[
            self.ratings['UserID_encoded'] == user_id
        ].sort_values('Rating', ascending=False)
        
        return user_ratings['ItemID_encoded'].tolist()[:max_items]
    
    def get_item_popularity(self) -> pd.Series:
        """
        Получение популярности предметов.
        
        Returns:
            Series с количеством рейтингов для каждого предмета
        """
        return self.ratings['ItemID_encoded'].value_counts()
