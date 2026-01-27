"""
Модель DeepFM+SVD++ для предсказания рейтингов.

Гибридная модель, объединяющая:
- Factorization Machine (FM) для моделирования попарных взаимодействий
- Deep Neural Network для нелинейных взаимодействий
- SVD++ для учета implicit feedback (истории пользователя)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple


class DeepFMSVDPlusPlus(nn.Module):
    """
    Гибридная модель DeepFM + SVD++ для предсказания многокритериальных рейтингов.
    
    Предсказывает 4 критерия:
    - Rating: Общий рейтинг
    - App: Предметная область
    - Data: Тип данных
    - Ease: Сложность
    
    Attributes:
        n_users: Количество пользователей
        n_items: Количество предметов
        n_classes: Количество классов (специализаций)
        n_semesters: Количество семестров
        n_lockdowns: Количество периодов COVID
        embedding_dim: Размерность эмбеддингов
        hidden_dims: Список размерностей скрытых слоев Deep части
        dropout_rate: Коэффициент dropout
        device: Устройство для вычислений (cuda/cpu)
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_classes: int,
        n_semesters: int,
        n_lockdowns: int,
        device: torch.device,
        embedding_dim: int = 32,
        hidden_dims: List[int] = [64, 32],
        dropout_rate: float = 0.2
    ):
        """
        Инициализация модели DeepFM+SVD++.
        
        Args:
            n_users: Количество пользователей
            n_items: Количество предметов
            n_classes: Количество классов (специализаций)
            n_semesters: Количество семестров
            n_lockdowns: Количество периодов COVID
            device: Устройство для вычислений
            embedding_dim: Размерность эмбеддингов (по умолчанию 32)
            hidden_dims: Список размерностей скрытых слоев (по умолчанию [64, 32])
            dropout_rate: Коэффициент dropout (по умолчанию 0.2)
        """
        super(DeepFMSVDPlusPlus, self).__init__()
        
        # Эмбеддинги для FM части
        self.user_emb_fm = nn.Embedding(n_users, embedding_dim)
        self.item_emb_fm = nn.Embedding(n_items, embedding_dim)
        self.class_emb_fm = nn.Embedding(n_classes, embedding_dim // 2)
        self.semester_emb_fm = nn.Embedding(n_semesters, embedding_dim // 4)
        self.lockdown_emb_fm = nn.Embedding(n_lockdowns, embedding_dim // 4)
        
        # Эмбеддинги для SVD++ (implicit feedback)
        self.user_emb_svd = nn.Embedding(n_users, embedding_dim)
        self.item_emb_svd = nn.Embedding(n_items, embedding_dim)
        
        # Linear bias terms
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Deep часть
        # Размеры эмбеддингов:
        # user_emb_fm: embedding_dim
        # item_emb_fm: embedding_dim
        # class_emb: embedding_dim // 2
        # semester_emb: embedding_dim // 4
        # lockdown_emb: embedding_dim // 4
        # user_emb_svd: embedding_dim
        # item_emb_svd: embedding_dim
        # ИТОГО: 4*embedding_dim + embedding_dim//2 + 2*(embedding_dim//4)
        total_embed_dim = embedding_dim * 4 + (embedding_dim // 2) + (embedding_dim // 4) * 2
        
        deep_layers = []
        deep_input_dim = total_embed_dim
        
        for hidden_dim in hidden_dims:
            deep_layers.append(nn.Linear(deep_input_dim, hidden_dim))
            deep_layers.append(nn.BatchNorm1d(hidden_dim))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.Dropout(dropout_rate))
            deep_input_dim = hidden_dim
        
        self.deep_network = nn.Sequential(*deep_layers)
        
        # Выходные слои для каждого критерия
        # Размер hidden_dims[-1] + 1 (fm_second_order) + 1 (linear_terms) = hidden_dims[-1] + 2
        self.rating_output = nn.Linear(hidden_dims[-1] + 2, 1)
        self.app_output = nn.Linear(hidden_dims[-1] + 2, 1)
        self.data_output = nn.Linear(hidden_dims[-1] + 2, 1)
        self.ease_output = nn.Linear(hidden_dims[-1] + 2, 1)

        self.device = device
        
        # Инициализация
        self._init_weights()
        
        # Перемещаем модель на устройство
        self.to(self.device)
    
    def _init_weights(self):
        """Инициализация весов модели."""
        for emb_layer in [self.user_emb_fm, self.item_emb_fm, self.user_emb_svd, self.item_emb_svd]:
            nn.init.xavier_uniform_(emb_layer.weight)
        
        for layer in self.deep_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        for output_layer in [self.rating_output, self.app_output, self.data_output, self.ease_output]:
            nn.init.xavier_uniform_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        class_ids: torch.Tensor,
        semester_ids: torch.Tensor,
        lockdown_ids: torch.Tensor,
        user_history: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Прямой проход через модель.
        
        Args:
            user_ids: Tensor с ID пользователей [batch_size]
            item_ids: Tensor с ID предметов [batch_size]
            class_ids: Tensor с ID классов [batch_size]
            semester_ids: Tensor с ID семестров [batch_size]
            lockdown_ids: Tensor с ID периодов COVID [batch_size]
            user_history: Опциональный Tensor с историей пользователя для SVD++
        
        Returns:
            Словарь с предсказаниями:
            {
                'rating': Tensor [batch_size],
                'app': Tensor [batch_size],
                'data': Tensor [batch_size],
                'ease': Tensor [batch_size]
            }
        """
        # FM часть: линейные и попарные взаимодействия
        user_emb_fm = self.user_emb_fm(user_ids)
        item_emb_fm = self.item_emb_fm(item_ids)
        class_emb = self.class_emb_fm(class_ids)
        semester_emb = self.semester_emb_fm(semester_ids)
        lockdown_emb = self.lockdown_emb_fm(lockdown_ids)
        
        # Линейные термины
        linear_terms = self.user_bias(user_ids).squeeze() + \
                      self.item_bias(item_ids).squeeze() + \
                      self.global_bias
        
        # Попарные взаимодействия FM
        # Формула: 0.5 * (sum_i sum_j <v_i, v_j> x_i x_j) = 
        # 0.5 * ((sum_i v_i x_i)^2 - sum_i (v_i^2 x_i^2))
        
        # Взаимодействия между user и item embeddings
        interaction_term = torch.sum(user_emb_fm * item_emb_fm, dim=1)  # [batch_size]
        
        # Квадрат суммы взаимодействий
        square_of_sum = torch.pow(interaction_term, 2)  # [batch_size]
        
        # Сумма квадратов
        sum_of_squares = torch.sum(torch.pow(user_emb_fm, 2) * torch.pow(item_emb_fm, 2), dim=1)  # [batch_size]
        
        # FM второго порядка
        fm_second_order = 0.5 * (square_of_sum - sum_of_squares)  # [batch_size]
        
        # SVD++ часть: implicit feedback
        if user_history is not None and len(user_history) > 0:
            # Усреднение эмбеддингов просмотренных items
            history_embs = self.item_emb_svd(user_history)
            history_mean = torch.mean(history_embs, dim=1)
            user_emb_svd = self.user_emb_svd(user_ids) + history_mean
        else:
            user_emb_svd = self.user_emb_svd(user_ids)
        
        item_emb_svd = self.item_emb_svd(item_ids)
        
        # Объединение всех эмбеддингов для deep части
        deep_input = torch.cat([
            user_emb_fm, item_emb_fm, 
            class_emb, semester_emb, lockdown_emb,
            user_emb_svd, item_emb_svd
        ], dim=1)
        
        # Deep часть
        deep_output = self.deep_network(deep_input)
        
        # Объединение FM и Deep для финального предсказания
        combined_features = torch.cat([
            deep_output, 
            fm_second_order.unsqueeze(1), 
            linear_terms.unsqueeze(1)
        ], dim=1)
        
        # Предсказания для каждого критерия
        rating_pred = torch.sigmoid(self.rating_output(combined_features))
        app_pred = torch.sigmoid(self.app_output(combined_features))
        data_pred = torch.sigmoid(self.data_output(combined_features))
        ease_pred = torch.sigmoid(self.ease_output(combined_features))
        
        return {
            'rating': rating_pred.squeeze(),
            'app': app_pred.squeeze(),
            'data': data_pred.squeeze(),
            'ease': ease_pred.squeeze()
        }
    
    def get_embeddings(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Получение эмбеддингов для RL агента.
        
        Args:
            user_ids: Tensor с ID пользователей [batch_size]
            item_ids: Tensor с ID предметов [batch_size]
        
        Returns:
            Tuple (user_embeddings, item_embeddings):
            - user_embeddings: Tensor [batch_size, embedding_dim * 2]
            - item_embeddings: Tensor [batch_size, embedding_dim * 2]
        """
        with torch.no_grad():
            user_emb = self.user_emb_fm(user_ids)
            item_emb = self.item_emb_fm(item_ids)
            
            # Конкатенация с SVD++ эмбеддингами
            user_emb_svd = self.user_emb_svd(user_ids)
            item_emb_svd = self.item_emb_svd(item_ids)
            
            user_final = torch.cat([user_emb, user_emb_svd], dim=1)
            item_final = torch.cat([item_emb, item_emb_svd], dim=1)
            
            return user_final, item_final
