"""
Приоритетный буфер воспроизведения опыта (Prioritized Experience Replay).

Реализует алгоритм PER для более эффективного обучения DQN,
где переходы с большей TD-ошибкой выбираются чаще.
"""

import numpy as np
import torch
from typing import Optional, Tuple, List


class PrioritizedReplayBuffer:
    """
    Буфер воспроизведения опыта с приоритетами.
    
    Использует Prioritized Experience Replay (PER) для более эффективного
    обучения за счет приоритизации переходов с высокой TD-ошибкой.
    
    Attributes:
        capacity: Максимальная емкость буфера
        alpha: Степень приоритизации (0 = равномерная выборка, 1 = полная приоритизация)
        beta: Степень коррекции смещения (importance sampling)
        buffer: Список переходов (state, action, reward, next_state, done)
        priorities: Список приоритетов для каждого перехода
        position: Текущая позиция в циклическом буфере
        beta_increment: Шаг увеличения beta до 1.0
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta: float = 0.4
    ):
        """
        Инициализация буфера.
        
        Args:
            capacity: Максимальная емкость буфера
            alpha: Степень приоритизации (0-1)
            beta: Начальная степень коррекции смещения (0-1)
        """
        self.capacity = capacity
        self.alpha = alpha  # Степень приоритизации (0 - равномерная выборка)
        self.beta = beta    # Степень коррекции смещения
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.beta_increment = 0.001
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Добавление перехода в буфер.
        
        Args:
            state: Текущее состояние
            action: Выполненное действие
            reward: Полученная награда
            next_state: Следующее состояние
            done: Флаг завершения эпизода
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            self.priorities.append(None)
        
        # Максимальный приоритет для новых переходов
        valid_priorities = [p for p in self.priorities if p is not None]
        max_priority = max(valid_priorities) if valid_priorities else 1.0
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(
        self,
        batch_size: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int], torch.Tensor]]:
        """
        Выборка батча с учетом приоритетов.
        
        Args:
            batch_size: Размер батча
        
        Returns:
            Tuple (states, actions, rewards, next_states, dones, indices, weights) или None
            - states: Tensor состояний [batch_size, state_dim]
            - actions: Tensor действий [batch_size]
            - rewards: Tensor наград [batch_size]
            - next_states: Tensor следующих состояний [batch_size, state_dim]
            - dones: Tensor флагов завершения [batch_size]
            - indices: Список индексов выбранных переходов
            - weights: Tensor весов importance sampling [batch_size]
        """
        if len(self.buffer) == 0:
            return None
        
        # Преобразование приоритетов в вероятности
        priorities = np.array(self.priorities[:len(self.buffer)])
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Выбор индексов
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Коррекция смещения (importance sampling weights)
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Обновление beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Сборка батча
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones),
            indices.tolist(),
            torch.FloatTensor(weights)
        )
    
    def update_priorities(self, indices: List[int], errors: np.ndarray):
        """
        Обновление приоритетов на основе ошибок TD.
        
        Args:
            indices: Индексы переходов в буфере
            errors: TD-ошибки для этих переходов
        """
        for idx, error in zip(indices, errors):
            self.priorities[idx] = abs(error) + 1e-6  # Добавление маленького значения
    
    def __len__(self) -> int:
        """Возвращает текущий размер буфера."""
        return len(self.buffer)
