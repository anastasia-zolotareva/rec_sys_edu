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
from .reward import (
    calculate_cosine_novelty,
    calculate_itmrec_reward,
)


class EducationalEnvironment:
    """Симулятор образовательной среды для обучения RL агента.

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
        dataset,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Инициализирует образовательную среду.

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
        
        # Параметры среды (могут быть переопределены через config)
        cfg = dict(config or {})
        env_cfg = cfg.get('environment', cfg)
        self.max_trajectory_length = int(env_cfg.get('max_trajectory_length', 10))
        self.max_recommendations = int(env_cfg.get('max_recommendations', 5))
        self.novelty_weight = float(env_cfg.get('novelty_weight', 0.05))
        self.reward_mode = env_cfg.get('reward_mode', 'itmrec')
        self.novelty_mode = env_cfg.get('novelty_mode', 'cosine')  # 'cosine' или 'popularity'
        self.reward_base_weights = env_cfg.get('reward_weights', None)
        # Порог низкой награды = 0.3, случайное завершение p=0.1
        self.low_reward_threshold = float(env_cfg.get('low_reward_threshold', 0.3))
        self.low_reward_window = int(env_cfg.get('low_reward_window', 3))
        self.enable_low_reward_termination = bool(
            env_cfg.get('enable_low_reward_termination', True)  # Включено по умолчанию
        )
        # Случайное завершение с вероятностью p
        self.termination_random_prob = float(env_cfg.get('termination_random_prob', 0.1))
        
        # Популярность предметов (для режима novelty='popularity' и fallback)
        self.item_popularity = dict(
            self.ratings['ItemID_encoded'].value_counts().to_dict()
        )
        self.max_item_popularity = max(self.item_popularity.values()) if self.item_popularity else 1
        
        # Инициализация кэшей
        self._initialize_caches()
    
    def _initialize_caches(self):
        """Инициализирует кэши эмбеддингов и истории пользователей."""
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
            self.user_history_cache[user_id] = user_history[:10]  # Последние 10 элементов
        
        print(f"Кэши инициализированы: {len(self.user_history_cache)} пользователей, "
              f"{len(self.item_embeddings_cache)} предметов")
    
    def reset(
        self,
        user_id: Optional[int] = None,
        context: Optional[Dict[str, int]] = None
    ) -> np.ndarray:
        """Инициализирует среду для нового эпизода.

        Args:
            user_id: ID пользователя (если None, выбирается случайно)
            context: Контекстные переменные (если None, выбирается из истории пользователя)

        Returns:
            Начальное состояние размерности 65
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
    
    # Сегменты состояния (см. _get_state ниже). Используются модулями H2
    # для ablation-анализа по конфигурациям состояния (``state_ablation``).
    STATE_SEGMENTS = {
        "user": (0, 32),
        "context": (32, 42),
        "demo": (42, 48),
        "history": (48, 63),
        "time": (63, 65),
    }

    def _apply_state_ablation(self, state: np.ndarray) -> np.ndarray:
        """Зануляет отключенные сегменты состояния согласно ``self.state_ablation``.

        Поддерживаемые режимы: ``None`` / ``"full"`` (без ablation),
        ``"no_context"``, ``"no_demo"``, ``"no_history"``,
        ``"no_context_no_demo"``.
        """
        mode = getattr(self, "state_ablation", None)
        if not mode or mode == "full":
            return state
        segments_to_zero: List[str] = []
        if mode == "no_context":
            segments_to_zero = ["context"]
        elif mode == "no_demo":
            segments_to_zero = ["demo"]
        elif mode == "no_history":
            segments_to_zero = ["history"]
        elif mode == "no_context_no_demo":
            segments_to_zero = ["context", "demo"]
        else:
            return state
        state = state.copy()
        for name in segments_to_zero:
            lo, hi = self.STATE_SEGMENTS[name]
            state[lo:hi] = 0.0
        return state

    def _get_state(self) -> np.ndarray:
        """Формирует вектор состояния размерности 65.

        Returns:
            Вектор состояния
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

        return self._apply_state_ablation(state)
    
    def _get_user_embedding(self, user_id: int) -> np.ndarray:
        """Вычисляет эмбеддинг пользователя.

        Args:
            user_id: Закодированный ID пользователя

        Returns:
            Эмбеддинг пользователя размерности 32
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
        """Выполняет один шаг в среде.

        Args:
            action: ID предмета для рекомендации (0 до n_items-1)

        Returns:
            Tuple (next_state, reward, done, info):
            - next_state: Следующее состояние размерности 65
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
        """Имитирует реакцию пользователя на рекомендацию.

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
        """Вычисляет вознаграждение по многокритериальной формуле ITM-Rec.

        Использует функцию calculate_itmrec_reward из модуля reward.
        """
        demo_vector = get_demographic_vector(
            self.current_user,
            self.users,
            self.dataset.user_encoder,
        )
        novelty = self._calculate_novelty(action)
        return calculate_itmrec_reward(
            feedback=feedback,
            context=self.current_context,
            demo_vector=demo_vector,
            novelty=novelty,
            novelty_weight=self.novelty_weight,
            base_weights=self.reward_base_weights,
        )
    
    def _calculate_novelty(self, action: int) -> float:
        """Вычисляет новизну рекомендации.

        - ``cosine``: 1 минус максимальное косинусное сходство с уже рекомендованными.
        - ``popularity``: 1 минус нормализованная популярность.
        """
        if self.novelty_mode == 'cosine':
            novelty = calculate_cosine_novelty(
                action=action,
                recommended_items=self.recommended_items,
                item_embeddings_cache=self.item_embeddings_cache,
                popularity=self.item_popularity,
                max_popularity=self.max_item_popularity,
            )
        else:
            pop = self.item_popularity.get(action, 0)
            novelty = 1.0 - pop / max(self.max_item_popularity, 1)
        
        if action not in self.recommended_items:
            novelty += 0.1
        return float(np.clip(novelty, 0.0, 1.0))
    
    def _check_termination(self) -> bool:
        """Проверяет условия завершения эпизода.

        Завершение происходит при:
        1. Достигнута максимальная длина траектории
        2. Достигнуто максимальное количество рекомендаций
        3. Средняя награда последних N шагов ниже порога (0.3)
        4. Случайное завершение с вероятностью p=0.1
        """
        import random
        
        # Случайное завершение с вероятностью p
        if random.random() < self.termination_random_prob:
            return True
        
        if self.step_count >= self.max_trajectory_length:
            return True
        if len(self.recommended_items) >= self.max_recommendations:
            return True
        if self.enable_low_reward_termination and len(self.trajectory) >= self.low_reward_window:
            recent_rewards = [r['reward'] for r in self.trajectory[-self.low_reward_window:]]
            if np.mean(recent_rewards) < self.low_reward_threshold:
                return True
        return False

    def get_action_mask(self) -> np.ndarray:
        """Создает маску допустимых действий для ITM-Rec.

        Для ITM-Rec все предметы валидны, но маскируются уже рекомендованные,
        чтобы агент не выбирал их повторно.
        """
        mask = np.ones(self.dataset.n_items, dtype=np.float32)
        for item_id in self.recommended_items:
            if 0 <= int(item_id) < mask.shape[0]:
                mask[int(item_id)] = 0.0
        if mask.sum() == 0:
            mask[:] = 1.0
        return mask
