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
        self.hidden_dims = list(hidden_dims)
        
        # Device tracking - defined before creating layers
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Common feature extractor
        self.feature_layers = nn.ModuleList()
        input_dim = state_dim
        
        for hidden_dim in self.hidden_dims:
            self.feature_layers.append(nn.Linear(input_dim, hidden_dim))
            self.feature_layers.append(nn.BatchNorm1d(hidden_dim))
            self.feature_layers.append(nn.ReLU())
            self.feature_layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        # Dueling DQN architecture
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
        # Transfer model to device after weight initialization
        self.to(self.device)
    
    def _init_weights(self):
        """Initialize model weights."""
        for layer in self.feature_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(layer.bias)
        
        for layer in [self.value_stream, self.advantage_stream]:
            for sublayer in layer:
                if isinstance(sublayer, nn.Linear):
                    nn.init.xavier_uniform_(sublayer.weight)
                    nn.init.zeros_(sublayer.bias)
    
    def forward(
        self,
        state: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor [batch_size, state_dim]
            action_mask: Binary mask of valid actions [batch_size, action_dim] or
                [action_dim]. Invalid actions will receive Q = -inf.
        
        Returns:
            Q-values for all actions [batch_size, action_dim]
        """
        features = state
        
        for i in range(0, len(self.feature_layers), 4):
            linear, bn, relu, dropout = self.feature_layers[i:i+4]
            features = linear(features)
            if features.size(0) > 1:  # BatchNorm requires batch_size > 1
                features = bn(features)
            features = relu(features)
            features = dropout(features)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        if action_mask is not None:
            mask = action_mask
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand_as(q_values)
            mask = mask.to(dtype=torch.bool, device=q_values.device)
            q_values = q_values.masked_fill(~mask, float('-inf'))

        return q_values
    
    def get_action(
        self,
        state: np.ndarray,
        epsilon: float = 0.1,
        action_mask: Optional[np.ndarray] = None,
    ) -> int:
        """
        Action selection using epsilon-greedy strategy.
        
        Args:
            state: State as numpy array [state_dim]
            epsilon: Probability of random action (0.0 - greedy, 1.0 - random)
            action_mask: Optional binary mask of valid actions [action_dim].
                Only actions where mask = 1 are selected.
        
        Returns:
            Index of selected action
        """
        if action_mask is not None:
            valid_actions = np.flatnonzero(np.asarray(action_mask) > 0)
            if valid_actions.size == 0:
                return int(np.random.randint(0, self.action_dim))
        else:
            valid_actions = None

        if np.random.random() < epsilon:
            if valid_actions is not None:
                return int(np.random.choice(valid_actions))
            return int(np.random.randint(0, self.action_dim))

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            mask_tensor = None
            if action_mask is not None:
                mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
            q_values = self.forward(state_tensor, mask_tensor)
            return int(q_values.argmax(dim=1).item())
    
    def predict_q_values(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Predict Q-values for a batch of states.
        
        Args:
            state_batch: Batch of states [batch_size, state_dim]
        
        Returns:
            Q-values for all actions [batch_size, action_dim]
        """
        with torch.no_grad():
            return self.forward(state_batch)
