"""
Модель Dueling DQN для обучения с подкреплением.

Dueling DQN архитектура разделяет оценку состояния на:
- Value stream: V(s) - ценность состояния
- Advantage stream: A(s,a) - преимущество действия
- Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional


class DuelingDQN(nn.Module):
    """
    Архитектура Dueling DQN для рекомендательной системы.
    
    Dueling DQN разделяет оценку Q-значений на два потока:
    - Value stream: оценивает ценность состояния V(s)
    - Advantage stream: оценивает преимущество действий A(s,a)
    - Финальное Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
    
    Это позволяет агенту лучше оценивать состояния независимо от действий.
    
    Attributes:
        state_dim: Размерность состояния
        action_dim: Размерность пространства действий (количество предметов)
        hidden_dims: Список размерностей скрытых слоев
        device: Устройство для вычислений (cuda/cpu)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        device: Optional[torch.device] = None
    ):
        """
        Инициализация Dueling DQN.
        
        Args:
            state_dim: Размерность состояния
            action_dim: Размерность пространства действий
            hidden_dims: Список размерностей скрытых слоев (по умолчанию [256, 128, 64])
            device: Устройство для вычислений (если None, определяется автоматически)
        """
        super(DuelingDQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Device tracking - определяем до создания слоев
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Общий экстрактор признаков
        self.feature_layers = nn.ModuleList()
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            self.feature_layers.append(nn.Linear(input_dim, hidden_dim))
            self.feature_layers.append(nn.BatchNorm1d(hidden_dim))
            self.feature_layers.append(nn.ReLU())
            self.feature_layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        # Dueling архитектура
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        
        # Инициализация весов
        self._init_weights()
        
        # Перемещаем модель на устройство после инициализации весов
        self.to(self.device)
    
    def _init_weights(self):
        """Инициализация весов модели."""
        for layer in self.feature_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(layer.bias)
        
        for layer in [self.value_stream, self.advantage_stream]:
            for sublayer in layer:
                if isinstance(sublayer, nn.Linear):
                    nn.init.xavier_uniform_(sublayer.weight)
                    nn.init.zeros_(sublayer.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через сеть.
        
        Args:
            state: Tensor состояния [batch_size, state_dim]
        
        Returns:
            Q-значения для всех действий [batch_size, action_dim]
        """
        # Убедимся, что вход уже на правильном устройстве
        features = state
        
        # Общий экстрактор признаков
        for i in range(0, len(self.feature_layers), 4):
            linear, bn, relu, dropout = self.feature_layers[i:i+4]
            features = linear(features)
            if features.size(0) > 1:  # BatchNorm требует batch_size > 1
                features = bn(features)
            features = relu(features)
            features = dropout(features)
        
        # Value и Advantage потоки
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Объединение по формуле Dueling DQN
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values
    
    def get_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """
        Выбор действия с использованием ε-жадной стратегии.
        
        Args:
            state: Состояние как numpy array [state_dim]
            epsilon: Вероятность случайного действия (0.0 - жадное, 1.0 - случайное)
        
        Returns:
            Индекс выбранного действия
        """
        if np.random.random() < epsilon:
            # Случайное действие
            return np.random.randint(0, self.action_dim)
        else:
            # Жадное действие
            with torch.no_grad():
                # Перемещаем состояние на то же устройство, что и модель
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.forward(state_tensor)
                return q_values.argmax().item()
    
    def predict_q_values(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Предсказание Q-значений для батча состояний.
        
        Args:
            state_batch: Батч состояний [batch_size, state_dim]
        
        Returns:
            Q-значения для всех действий [batch_size, action_dim]
        """
        with torch.no_grad():
            return self.forward(state_batch)
