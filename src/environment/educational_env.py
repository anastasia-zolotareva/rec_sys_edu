"""
Симулятор образовательной среды для обучения RL агента.

EducationalEnvironment реализует интерфейс OpenAI Gym для обучения
DQN агента в задаче рекомендаций образовательных тем проектов.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, Optional, List, Tuple, Any

from .state_encoder import encode_context, encode_history, get_demographic_vector


class EducationalEnvironment:
    """
    Симулятор образовательной среды для RL обучения.
    
    Состояние (state):
    - Эмбеддинг пользователя (32-dim)
    - Контекстный вектор (10-dim)
    - Демографический вектор (6-dim)
    - История взаимодействий (15-dim)
    - Временные метки (2-dim)
    ИТОГО: 65 измерений
    
    Действие (action):
    - ID рекомендованного предмета (0 до n_items-1)
    
    Награда (reward):
    - Многокритериальная: Rating, App, Data, Ease
    - С учетом новизны (novelty bonus)
    - Контекстно-зависимая
    """
    
    def __init__(
        self,
        ratings_df: pd.DataFrame,
        users_df: pd.DataFrame,
        items_df: pd.DataFrame,
        deepfm_model,
        dataset
    ):
        """
        Инициализация среды.
        
        Args:
            ratings_df: DataFrame с рейтингами
            users_df: DataFrame с пользователями
            items_df: DataFrame с предметами
            deepfm_model: Обученная модель DeepFM+SVD++
            dataset: ITMDataset с кодировщиками
        """
        # Используем dataset.ratings, который содержит закодированные колонки
        self.ratings = dataset.ratings.copy()
        self.users = users_df.copy()
        self.items = items_df.copy()
        self.model = deepfm_model
        self.dataset = dataset
        
        # Определение устройства
        self.device = next(self.model.parameters()).device
        
        # Кэширование
        self.user_history_cache = {}
        self.item_embeddings_cache = {}
        self.user_embeddings_cache = {}
        
        # Параметры среды
        self.max_trajectory_length = 10
        self.max_recommendations = 5
        self.novelty_weight = 0.05
        
        # Инициализация кэшей
        self._initialize_caches()
    
    def _initialize_caches(self):
        """Инициализация кэшей для ускорения работы."""
        print("Инициализация кэшей...")
        
        # Кэш эмбеддингов предметов
        all_item_ids = torch.arange(self.dataset.n_items).long().to(self.device)
        with torch.no_grad():
            item_embeddings = self.model.item_emb_fm(all_item_ids)
            for i, item_id in enumerate(all_item_ids.cpu().numpy()):
                self.item_embeddings_cache[item_id] = item_embeddings[i].cpu().numpy()
        
        # Кэш истории пользователей
        for user_id in self.ratings['UserID_encoded'].unique():
            user_history = self.ratings[
                self.ratings['UserID_encoded'] == user_id
            ]['ItemID_encoded'].tolist()
            self.user_history_cache[user_id] = user_history[:10]  # Последние 10
        
        print(f"Кэши инициализированы: {len(self.user_history_cache)} пользователей, "
              f"{len(self.item_embeddings_cache)} предметов")
    
    def reset(
        self,
        user_id: Optional[int] = None,
        context: Optional[Dict[str, int]] = None
    ) -> np.ndarray:
        """
        Сброс среды для нового эпизода.
        
        Args:
            user_id: ID пользователя (если None, выбирается случайно)
            context: Контекстные переменные (если None, выбирается из истории пользователя)
        
        Returns:
            Начальное состояние [65]
        """
        if user_id is None:
            # Случайный выбор пользователя
            self.current_user = np.random.choice(self.ratings['UserID_encoded'].unique())
        else:
            self.current_user = user_id
        
        # Установка контекста
        if context is None:
            # Случайный контекст на основе истории пользователя
            user_data = self.ratings[self.ratings['UserID_encoded'] == self.current_user]
            if len(user_data) > 0:
                sample = user_data.iloc[0]
                self.current_context = {
                    'class': int(sample['Class_encoded']),
                    'semester': int(sample['Semester_encoded']),
                    'lockdown': int(sample['Lockdown_encoded'])
                }
            else:
                # Контекст по умолчанию
                self.current_context = {
                    'class': 0,  # DA
                    'semester': 0,  # Fall
                    'lockdown': 0  # PRE
                }
        else:
            self.current_context = context
        
        # Сброс состояния эпизода
        self.trajectory = []
        self.recommended_items = set()
        self.step_count = 0
        self.cumulative_reward = 0.0
        
        # Получение начального состояния
        initial_state = self._get_state()
        
        print(f"Эпизод начат: User {self.current_user}, Context {self.current_context}")
        
        return initial_state
    
    def _get_state(self) -> np.ndarray:
        """
        Формирование вектора состояния s_t.
        
        Returns:
            Вектор состояния размерности 65
        """
        state_components = []
        
        # 1. User Embedding (32-dim)
        user_emb = self._get_user_embedding(self.current_user)
        state_components.append(user_emb)
        
        # 2. Контекстный вектор (10-dim)
        context_vector = encode_context(self.current_context, self.trajectory)
        state_components.append(context_vector)
        
        # 3. Демографический вектор (6-dim)
        demo_vector = get_demographic_vector(
            self.current_user,
            self.users,
            self.dataset.user_encoder
        )
        state_components.append(demo_vector)
        
        # 4. История взаимодействий (15-dim)
        history_vector = encode_history(self.trajectory, self.item_embeddings_cache)
        state_components.append(history_vector)
        
        # 5. Временные метки (2-dim)
        time_vector = np.array([
            self.step_count / self.max_trajectory_length,  # Прогресс эпизода
            len(self.trajectory) / 10.0  # Длина истории
        ])
        state_components.append(time_vector)
        
        # Объединение всех компонентов
        state = np.concatenate(state_components, axis=0)
        
        # Проверка размерности
        expected_dim = 32 + 10 + 6 + 15 + 2  # 65
        assert len(state) == expected_dim, f"State dimension mismatch: {len(state)} != {expected_dim}"
        
        return state
    
    def _get_user_embedding(self, user_id: int) -> np.ndarray:
        """
        Получение эмбеддинга пользователя.
        
        Args:
            user_id: Закодированный ID пользователя
        
        Returns:
            Эмбеддинг пользователя [32]
        """
        if user_id in self.user_embeddings_cache:
            return self.user_embeddings_cache[user_id]
        
        # Вычисление эмбеддинга через модель
        user_tensor = torch.LongTensor([user_id]).to(self.device)
        with torch.no_grad():
            user_emb = self.model.user_emb_fm(user_tensor).cpu().numpy().flatten()
        
        self.user_embeddings_cache[user_id] = user_emb
        return user_emb
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Выполнение шага в среде.
        
        Args:
            action: ID предмета для рекомендации (0 до n_items-1)
        
        Returns:
            Tuple (next_state, reward, done, info):
            - next_state: Следующее состояние [65]
            - reward: Награда за действие
            - done: Флаг завершения эпизода
            - info: Дополнительная информация
        """
        self.step_count += 1
        
        # Проверка валидности действия
        if action < 0 or action >= self.dataset.n_items:
            raise ValueError(f"Invalid action: {action}")
        
        # Добавление в рекомендованные
        self.recommended_items.add(action)
        
        # Имитация фидбека от пользователя
        feedback = self._simulate_feedback(action)
        
        # Расчет вознаграждения
        reward = self._calculate_reward(feedback, action)
        self.cumulative_reward += reward
        
        # Обновление траектории
        interaction_record = {
            'item_id': action,
            'rating': feedback['rating'],
            'app': feedback['app'],
            'data': feedback['data'],
            'ease': feedback['ease'],
            'reward': reward,
            'time_since': 1.0 / (self.step_count + 1)  # Уменьшение со временем
        }
        self.trajectory.append(interaction_record)
        
        # Проверка завершения эпизода
        done = self._check_termination()
        
        # Получение следующего состояния
        if not done:
            next_state = self._get_state()
        else:
            # Вместо None возвращаем нулевой вектор состояния
            next_state = np.zeros(65)  # Размерность состояния = 65
        
        # Информация для отладки
        info = {
            'user_id': self.current_user,
            'item_id': action,
            'feedback': feedback,
            'cumulative_reward': self.cumulative_reward,
            'step_count': self.step_count
        }
        
        return next_state, reward, done, info
    
    def _simulate_feedback(self, item_id: int) -> Dict[str, float]:
        """
        Имитация реакции пользователя на рекомендацию.
        
        Args:
            item_id: ID рекомендованного предмета
        
        Returns:
            Словарь с оценками:
            {
                'rating': Общий рейтинг (1-5),
                'app': Предметная область (1-5),
                'data': Тип данных (1-5),
                'ease': Сложность (1-5)
            }
        """
        # Поиск реальной оценки, если есть
        user_ratings = self.ratings[
            (self.ratings['UserID_encoded'] == self.current_user) & 
            (self.ratings['ItemID_encoded'] == item_id)
        ]
        
        if not user_ratings.empty:
            # Используем медианную оценку
            feedback = {
                'rating': float(user_ratings['Rating'].median()),
                'app': float(user_ratings['App'].median()),
                'data': float(user_ratings['Data'].median()),
                'ease': float(user_ratings['Ease'].median())
            }
        else:
            # Предсказание через модель
            with torch.no_grad():
                user_tensor = torch.LongTensor([self.current_user]).to(self.device)
                item_tensor = torch.LongTensor([item_id]).to(self.device)
                class_tensor = torch.LongTensor([self.current_context['class']]).to(self.device)
                semester_tensor = torch.LongTensor([self.current_context['semester']]).to(self.device)
                lockdown_tensor = torch.LongTensor([self.current_context['lockdown']]).to(self.device)
                
                # Получение истории пользователя
                user_history = self.user_history_cache.get(self.current_user, [])
                if user_history:
                    history_tensor = torch.LongTensor(user_history[:5]).unsqueeze(0).to(self.device)
                else:
                    history_tensor = None
                
                # Предсказание
                predictions = self.model(
                    user_tensor, item_tensor, class_tensor, 
                    semester_tensor, lockdown_tensor, history_tensor
                )
                
                # Перенос на CPU для получения значений
                feedback = {
                    'rating': predictions['rating'].cpu().item() * 5.0,
                    'app': predictions['app'].cpu().item() * 5.0,
                    'data': predictions['data'].cpu().item() * 5.0,
                    'ease': predictions['ease'].cpu().item() * 5.0
                }
        
        # Ограничение значений в диапазоне [1, 5]
        for key in feedback:
            feedback[key] = np.clip(feedback[key], 1.0, 5.0)
        
        return feedback
    
    def _calculate_reward(self, feedback: Dict[str, float], action: int) -> float:
        """
        Расчет вознаграждения по многокритериальной формуле.
        
        Args:
            feedback: Словарь с оценками пользователя
            action: ID рекомендованного предмета
        
        Returns:
            Награда за действие
        """
        # Базовые веса из EDA
        weights = {'w1': 0.50, 'w2': 0.30, 'w3': 0.15, 'w4': 0.05}
        
        # Корректировка весов по контексту
        if self.current_context['lockdown'] in [1, 2]:  # DUR или POS
            weights['w3'] = 0.25  # Усиление важности Ease
            weights['w1'] = 0.45
        
        if self.current_context['class'] == 1:  # DB
            weights['w2'] = 0.35  # Усиление важности Data
        
        # Демографический множитель
        demo_vector = get_demographic_vector(
            self.current_user,
            self.users,
            self.dataset.user_encoder
        )
        married = demo_vector[-1]
        age_group = np.argmax(demo_vector[1:5])
        
        demo_multiplier = 1.0
        if married == 1:
            demo_multiplier *= 0.9  # Женатые более критичны
        
        if age_group in [0, 1]:  # <20 или 20-25
            demo_multiplier *= 1.1  # Молодые ценят больше
        
        # Расчет новизны
        novelty = self._calculate_novelty(action)
        
        # Базовое вознаграждение
        normalized_feedback = {
            'app': feedback['app'] / 5.0,
            'data': feedback['data'] / 5.0,
            'ease': feedback['ease'] / 5.0
        }
        
        base_reward = (
            weights['w1'] * normalized_feedback['app'] +
            weights['w2'] * normalized_feedback['data'] +
            weights['w3'] * normalized_feedback['ease']
        )
        
        # Финальное вознаграждение с учетом новизны и демографии
        reward = base_reward * demo_multiplier + self.novelty_weight * novelty
        
        return float(reward)
    
    def _calculate_novelty(self, action: int) -> float:
        """
        Расчет новизны рекомендации.
        
        Args:
            action: ID рекомендованного предмета
        
        Returns:
            Новизна (0-1), где 1 - максимальная новизна
        """
        # Новизна = обратная популярность предмета
        item_popularity = self.ratings[
            self.ratings['ItemID_encoded'] == action
        ].shape[0]
        
        max_popularity = self.ratings['ItemID_encoded'].value_counts().max()
        novelty = 1.0 - (item_popularity / max_popularity) if max_popularity > 0 else 1.0
        
        # Бонус за разнообразие (не рекомендовали ранее в эпизоде)
        if action not in self.recommended_items:
            novelty += 0.1
        
        return np.clip(novelty, 0.0, 1.0)
    
    def _check_termination(self) -> bool:
        """
        Проверка условий завершения эпизода.
        
        Returns:
            True если эпизод завершен
        """
        # Завершение по достижению максимальной длины траектории
        if self.step_count >= self.max_trajectory_length:
            return True
        
        # Завершение по достижению максимального количества рекомендаций
        if len(self.recommended_items) >= self.max_recommendations:
            return True
        
        # Завершение по низкой награде (опционально)
        if len(self.trajectory) > 0:
            recent_rewards = [r['reward'] for r in self.trajectory[-3:]]
            if len(recent_rewards) == 3 and np.mean(recent_rewards) < 0.2:
                return True
        
        return False
