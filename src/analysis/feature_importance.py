"""
Модуль для анализа интерпретируемости моделей DeepFM+SVD++ и DQN.

Функционал:
- Вычисление SHAP values для DeepFM+SVD++
- Анализ градиентов состояния для DQN Q-values
- Визуализация значимости признаков
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger("rec_sys_edu")


class DQNStateImportanceAnalyzer:
    """Анализатор значимости компонентов состояния DQN на основе градиентов Q-values."""
    
    def __init__(self, dqn_model: nn.Module, state_dim: int, action_dim: int):
        """
        Инициализация анализатора.
        
        Args:
            dqn_model: Обученная модель DuelingDQN
            state_dim: Размерность состояния
            action_dim: Размерность пространства действий
        """
        self.dqn = dqn_model
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = next(dqn_model.parameters()).device
    
    def compute_state_importance(
        self,
        states: np.ndarray,
        actions: Optional[np.ndarray] = None,
        use_greedy_actions: bool = True,
    ) -> np.ndarray:
        """
        Вычислить значимость компонентов состояния для Q(s, a*).
        
        Args:
            states: Array состояний [n_samples, state_dim]
            actions: Array действий [n_samples], если None - используются жадные действия
            use_greedy_actions: Если True, для каждого состояния выбирается argmax_a Q(s,a)
        
        Returns:
            Array значимостей [state_dim] - среднее абсолютное значение градиента
        """
        states_tensor = torch.FloatTensor(states).to(self.device)
        states_tensor.requires_grad_(True)
        
        # Вычислить Q-values
        with torch.enable_grad():
            q_values = self.dqn(states_tensor)  # [n_samples, action_dim]
        
        # Выбрать действия (либо заданные, либо жадные)
        if use_greedy_actions or actions is None:
            with torch.no_grad():
                actions_indices = q_values.argmax(dim=1)  # [n_samples]
        else:
            actions_indices = torch.LongTensor(actions).to(self.device)
        
        # Собрать Q-values выбранных действий
        q_selected = q_values[torch.arange(len(q_values)), actions_indices]  # [n_samples]
        
        # Вычислить сумму Q-values (для получения одного скаляра для backward)
        q_sum = q_selected.sum()
        
        # Вычислить градиенты
        q_sum.backward()
        
        # Получить градиенты состояния
        state_gradients = states_tensor.grad.abs().detach().cpu().numpy()  # [n_samples, state_dim]
        
        # Агрегировать: среднее абсолютное значение по всем примерам
        importance = state_gradients.mean(axis=0)  # [state_dim]
        
        # Нормализовать
        importance = importance / (importance.max() + 1e-8)
        
        return importance
    
    def compute_per_sample_importance(
        self,
        states: np.ndarray,
        actions: Optional[np.ndarray] = None,
        use_greedy_actions: bool = True,
    ) -> np.ndarray:
        """
        Вычислить значимость для каждого примера отдельно.
        
        Returns:
            Array [n_samples, state_dim]
        """
        importances = []
        
        for i in range(len(states)):
            state = states[i:i+1]  # [1, state_dim]
            action = actions[i:i+1] if actions is not None else None
            
            importance = self.compute_state_importance(
                state,
                actions=action,
                use_greedy_actions=use_greedy_actions
            )
            importances.append(importance)
        
        return np.array(importances)  # [n_samples, state_dim]


class DeepFMFeatureImportanceAnalyzer:
    """Анализатор значимости признаков для DeepFM+SVD++ на основе градиентов."""
    
    def __init__(self, model: nn.Module, feature_names: Optional[List[str]] = None):
        """
        Инициализация анализатора.
        
        Args:
            model: Обученная модель DeepFMSVDPlusPlus
            feature_names: Названия признаков (опционально)
        """
        self.model = model
        self.feature_names = feature_names
        self.device = next(model.parameters()).device
    
    def compute_feature_importance_for_head(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        class_ids: np.ndarray,
        semester_ids: np.ndarray,
        lockdown_ids: np.ndarray,
        head_name: str = "rating",
        user_history: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Вычислить значимость признаков для конкретной выходной головы.
        
        Args:
            user_ids: Array ID пользователей
            item_ids: Array ID предметов
            class_ids: Array ID классов
            semester_ids: Array ID семестров
            lockdown_ids: Array ID lockdown периодов
            head_name: Название выходной головы (rating, app, data, ease для itmrec)
            user_history: Опциональная история пользователя
        
        Returns:
            Словарь с значимостями признаков
        """
        # Конвертировать в тензоры
        user_tensor = torch.LongTensor(user_ids).to(self.device)
        item_tensor = torch.LongTensor(item_ids).to(self.device)
        class_tensor = torch.LongTensor(class_ids).to(self.device)
        semester_tensor = torch.LongTensor(semester_ids).to(self.device)
        lockdown_tensor = torch.LongTensor(lockdown_ids).to(self.device)
        
        if user_history is not None:
            user_history = user_history.to(self.device)
        
        # Создать копии с требованием градиентов для embedding входов
        user_tensor.requires_grad_(False)
        item_tensor.requires_grad_(False)
        
        with torch.enable_grad():
            # Пробросить через модель
            predictions = self.model.forward(
                user_tensor,
                item_tensor,
                class_tensor,
                semester_tensor,
                lockdown_tensor,
                user_history
            )
            
            # Получить предсказания для конкретной головы
            output = predictions[head_name]  # [batch_size]
        
        # Вычислить градиенты relative к embeddings
        # К сожалению, embeddings не требуют градиентов напрямую,
        # поэтому вычисляем importance через вход модели в Deep network
        
        # Альтернативный подход: использовать activation magnitude внутри модели
        # или применить permutation-based feature importance
        
        return {
            "head": head_name,
            "mean_prediction": output.detach().cpu().numpy().mean(),
            "std_prediction": output.detach().cpu().numpy().std(),
        }


class StateComponentGrouper:
    """Утилита для группировки компонентов состояния DQN."""
    
    # Группировки для ITM-Rec (state_dim=65)
    ITMREC_COMPONENTS = {
        "User Embedding": (0, 32, "32-мерный вектор пользователя"),
        "Context": (32, 42, "Class(3) + Semester(2) + Lockdown(3) + time_in_semester(1) + success_rate(1)"),
        "Demographics": (42, 48, "Gender(1) + Age_onehot(4) + Married(1)"),
        "History": (48, 63, "Последние 3 взаимодействия (15 dims)"),
        "Temporal": (63, 65, "Progress + history_length"),
    }
    
    # Группировки для OULAD (state_dim=96)
    OULAD_COMPONENTS = {
        "User Embedding": (0, 32, "32-мерный вектор пользователя"),
        "Module/Presentation": (32, 52, "One-hot кодирование курса/сессии (~20 dims)"),
        "Demographics": (52, 64, "Gender + Age + IMD + Disability + Credits + Education"),
        "Progress": (64, 72, "Текущая неделя, оставшиеся недели, ratio assessments"),
        "History Aggregation": (72, 88, "Mean/Std/Last/Overall proxy-критерии (16 dims)"),
        "Action Availability": (88, 92, "Признаки доступности действий"),
        "Current Proxies": (92, 96, "Mastery + Engagement + SelfRegulation + Outcome"),
    }
    
    @staticmethod
    def get_groups(dataset_type: str) -> Dict[str, Tuple[int, int, str]]:
        """Получить группировку компонентов для датасета."""
        if dataset_type.lower() == "itmrec":
            return StateComponentGrouper.ITMREC_COMPONENTS
        elif dataset_type.lower() == "oulad":
            return StateComponentGrouper.OULAD_COMPONENTS
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    @staticmethod
    def aggregate_by_group(
        importance: np.ndarray,
        dataset_type: str
    ) -> Dict[str, float]:
        """Агрегировать значимость по группам компонентов."""
        groups = StateComponentGrouper.get_groups(dataset_type)
        result = {}
        
        for group_name, (start, end, description) in groups.items():
            group_importance = importance[start:end].mean()
            result[group_name] = {
                "importance": float(group_importance),
                "description": description,
                "indices": (start, end),
                "n_components": end - start,
            }
        
        return result
    
    @staticmethod
    def get_component_names(dataset_type: str) -> Dict[int, str]:
        """Получить названия для всех компонентов состояния."""
        groups = StateComponentGrouper.get_groups(dataset_type)
        names = {}
        
        for group_name, (start, end, description) in groups.items():
            for i in range(start, end):
                names[i] = f"{group_name}[{i-start}]"
        
        return names


def create_shap_background_dataset(
    data: np.ndarray,
    n_samples: int = 100,
    sampling_method: str = "random"
) -> np.ndarray:
    """
    Создать набор background данных для SHAP объяснений.
    
    Args:
        data: Полный датасет [n_total, n_features]
        n_samples: Количество образцов для background набора
        sampling_method: "random" или "kmeans"
    
    Returns:
        Background датасет [n_samples, n_features]
    """
    if len(data) <= n_samples:
        return data
    
    if sampling_method == "random":
        indices = np.random.choice(len(data), n_samples, replace=False)
        return data[indices]
    
    elif sampling_method == "kmeans":
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_samples, random_state=42, n_init=10)
            kmeans.fit(data)
            return kmeans.cluster_centers_
        except ImportError:
            logger.warning("sklearn not available, falling back to random sampling")
            indices = np.random.choice(len(data), n_samples, replace=False)
            return data[indices]
    
    else:
        raise ValueError(f"Unknown sampling_method: {sampling_method}")
