"""
Модель DeepFM+SVD++ для предсказания рейтингов.

Гибридная модель, объединяющая:
- Factorization Machine (FM) для моделирования попарных взаимодействий
- Deep Neural Network для нелинейных взаимодействий
- SVD++ для учета implicit feedback (истории пользователя)

Модель поддерживает два режима:
- ``dataset_type="itmrec"`` - 5 контекстных полей (class, semester, lockdown),
  4 выходные головы (Rating, App, Data, Ease).
- ``dataset_type="oulad"`` - 3 контекстных поля (module, presentation, step_type),
  4 выходные головы (Mastery, Engagement, SelfRegulation, Outcome).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


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
    
    # Определения выходных голов для каждого dataset_type.
    HEAD_NAMES = {
        "itmrec": ("rating", "app", "data", "ease"),
        "oulad": ("mastery", "engagement", "selfregulation", "outcome"),
    }

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
        dropout_rate: float = 0.2,
        dataset_type: str = "itmrec",
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

        dataset_type = dataset_type.lower()
        if dataset_type not in self.HEAD_NAMES:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")
        self.dataset_type = dataset_type

        self.n_users = n_users
        self.n_items = n_items
        self.n_classes = n_classes
        self.n_semesters = n_semesters
        self.n_lockdowns = n_lockdowns
        self.embedding_dim = embedding_dim
        self.hidden_dims = list(hidden_dims)
        self.dropout_rate = dropout_rate

        # Embeddings for FM component
        self.user_emb_fm = nn.Embedding(n_users, embedding_dim)
        self.item_emb_fm = nn.Embedding(n_items, embedding_dim)
        self.class_emb_fm = nn.Embedding(n_classes, embedding_dim // 2)
        self.semester_emb_fm = nn.Embedding(n_semesters, embedding_dim // 4)
        self.lockdown_emb_fm = nn.Embedding(n_lockdowns, embedding_dim // 4)
        
        # Embeddings for SVD++ (implicit feedback from user history)
        self.user_emb_svd = nn.Embedding(n_users, embedding_dim)
        self.item_emb_svd = nn.Embedding(n_items, embedding_dim)
        
        # Linear bias terms
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Deep network component
        # Embedding dimensions:
        # user_emb_fm: embedding_dim
        # item_emb_fm: embedding_dim
        # class_emb: embedding_dim // 2
        # semester_emb: embedding_dim // 4
        # lockdown_emb: embedding_dim // 4
        # user_emb_svd: embedding_dim
        # item_emb_svd: embedding_dim
        # TOTAL: 4*embedding_dim + embedding_dim//2 + 2*(embedding_dim//4)
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
        
        # Output layers for each criterion
        # Size hidden_dims[-1] + 1 (fm_second_order) + 1 (linear_terms) = hidden_dims[-1] + 2
        head_input = hidden_dims[-1] + 2
        self.output_heads = nn.ModuleDict(
            {name: nn.Linear(head_input, 1) for name in self.HEAD_NAMES[dataset_type]}
        )

        # Back-compat aliases for dataset_type='itmrec' (used externally)
        if dataset_type == "itmrec":
            self.rating_output = self.output_heads["rating"]
            self.app_output = self.output_heads["app"]
            self.data_output = self.output_heads["data"]
            self.ease_output = self.output_heads["ease"]

        self.device = device
        
        # Model initialization
        self._init_weights()
        
        # Transfer model to device
        self.to(self.device)
    
    def _init_weights(self):
        """Initialize model weights."""
        for emb_layer in [self.user_emb_fm, self.item_emb_fm, self.user_emb_svd, self.item_emb_svd]:
            nn.init.xavier_uniform_(emb_layer.weight)
        
        for layer in self.deep_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        for output_layer in self.output_heads.values():
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
        # FM component: linear and pairwise interactions
        user_emb_fm = self.user_emb_fm(user_ids)
        item_emb_fm = self.item_emb_fm(item_ids)
        class_emb = self.class_emb_fm(class_ids)
        semester_emb = self.semester_emb_fm(semester_ids)
        lockdown_emb = self.lockdown_emb_fm(lockdown_ids)
        
        # Linear terms
        linear_terms = self.user_bias(user_ids).squeeze() + \
                      self.item_bias(item_ids).squeeze() + \
                      self.global_bias
        
        # Pairwise interactions in FM
        # Formula: 0.5 * (sum_i sum_j <v_i, v_j> x_i x_j) = 
        # 0.5 * ((sum_i v_i x_i)^2 - sum_i (v_i^2 x_i^2))
        
        # Interactions between user and item embeddings
        interaction_term = torch.sum(user_emb_fm * item_emb_fm, dim=1)  # [batch_size]
        
        # Square of sum of interactions
        square_of_sum = torch.pow(interaction_term, 2)  # [batch_size]
        
        # Sum of squares
        sum_of_squares = torch.sum(torch.pow(user_emb_fm, 2) * torch.pow(item_emb_fm, 2), dim=1)  # [batch_size]
        
        # FM second order
        fm_second_order = 0.5 * (square_of_sum - sum_of_squares)  # [batch_size]
        
        # SVD++ component: implicit feedback from user history
        if user_history is not None and len(user_history) > 0:
            # Average embeddings of viewed items from history
            history_embs = self.item_emb_svd(user_history)
            history_mean = torch.mean(history_embs, dim=1)
            user_emb_svd = self.user_emb_svd(user_ids) + history_mean
        else:
            user_emb_svd = self.user_emb_svd(user_ids)
        
        item_emb_svd = self.item_emb_svd(item_ids)
        
        # Combine all embeddings for deep network component
        deep_input = torch.cat([
            user_emb_fm, item_emb_fm, 
            class_emb, semester_emb, lockdown_emb,
            user_emb_svd, item_emb_svd
        ], dim=1)
        
        # Deep network component (BatchNorm requires batch_size >= 2).
        if deep_input.size(0) < 2 and self.training:
            was_training = True
            self.deep_network.eval()
            deep_output = self.deep_network(deep_input)
            if was_training:
                self.deep_network.train()
        else:
            deep_output = self.deep_network(deep_input)

        # Combine FM and Deep for final prediction
        combined_features = torch.cat([
            deep_output,
            fm_second_order.unsqueeze(1),
            linear_terms.unsqueeze(1)
        ], dim=1)

        # Predictions for each output head
        predictions: Dict[str, torch.Tensor] = {}
        for name, head in self.output_heads.items():
            predictions[name] = torch.sigmoid(head(combined_features)).squeeze(-1)

        return predictions
    
    def get_embeddings(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract embeddings for RL agent.
        
        Args:
            user_ids: Tensor with user IDs [batch_size]
            item_ids: Tensor with item IDs [batch_size]
        
        Returns:
            Tuple (user_embeddings, item_embeddings):
            - user_embeddings: Tensor [batch_size, embedding_dim * 2]
            - item_embeddings: Tensor [batch_size, embedding_dim * 2]
        """
        with torch.no_grad():
            user_emb = self.user_emb_fm(user_ids)
            item_emb = self.item_emb_fm(item_ids)

            # Concatenate with SVD++ embeddings
            user_emb_svd = self.user_emb_svd(user_ids)
            item_emb_svd = self.item_emb_svd(item_ids)

            user_final = torch.cat([user_emb, user_emb_svd], dim=1)
            item_final = torch.cat([item_emb, item_emb_svd], dim=1)

            return user_final, item_final

    # ------------------------------------------------------------------
    # Batch prediction and serialization
    # ------------------------------------------------------------------

    def predict_batch(
        self,
        user_ids: Sequence[int],
        item_ids: Sequence[int],
        context: Dict[str, Sequence[int]],
        user_history: Optional[torch.Tensor] = None,
        to_numpy: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Predict multi-criteria ratings for a batch of (user, item, context).

        Args:
            user_ids: User IDs (encoded).
            item_ids: Item IDs (encoded).
            context: Dictionary with keys ``class_ids``, ``semester_ids``, ``lockdown_ids``
                (for ``dataset_type='itmrec'``). For OULAD mode, keys are the same,
                but contain already encoded module/presentation/step-type values.
            user_history: Optional tensor of history [batch, seq].
            to_numpy: Return numpy arrays instead of tensors.
        """
        self.eval()
        device = self.device

        def _to_long(x: Sequence[int]) -> torch.Tensor:
            if isinstance(x, torch.Tensor):
                return x.long().to(device)
            return torch.as_tensor(x, dtype=torch.long, device=device)

        user_tensor = _to_long(user_ids)
        item_tensor = _to_long(item_ids)
        class_tensor = _to_long(context.get("class_ids", [0] * len(user_tensor)))
        semester_tensor = _to_long(context.get("semester_ids", [0] * len(user_tensor)))
        lockdown_tensor = _to_long(context.get("lockdown_ids", [0] * len(user_tensor)))

        if user_history is not None and not isinstance(user_history, torch.Tensor):
            user_history = torch.as_tensor(user_history, dtype=torch.long, device=device)

        with torch.no_grad():
            preds = self.forward(
                user_tensor,
                item_tensor,
                class_tensor,
                semester_tensor,
                lockdown_tensor,
                user_history,
            )

        if to_numpy:
            return {k: v.detach().cpu().numpy() for k, v in preds.items()}
        return preds

    def save_checkpoint(
        self,
        filepath: Union[str, "Path"],
        extra: Optional[Dict] = None,
    ) -> None:
        """Save model weights along with hyperparameters and additional objects."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_class": self.__class__.__name__,
            "hparams": {
                "n_users": self.n_users,
                "n_items": self.n_items,
                "n_classes": self.n_classes,
                "n_semesters": self.n_semesters,
                "n_lockdowns": self.n_lockdowns,
                "embedding_dim": self.embedding_dim,
                "hidden_dims": self.hidden_dims,
                "dropout_rate": self.dropout_rate,
                "dataset_type": self.dataset_type,
            },
        }
        if extra:
            checkpoint.update(extra)
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(
        cls,
        filepath: Union[str, "Path"],
        device: Optional[torch.device] = None,
    ) -> Tuple["DeepFMSVDPlusPlus", Dict]:
        """Load model and return (model, checkpoint_dict).

        ``checkpoint_dict`` contains original data (in particular, encoders/scalers,
        if they were saved by the caller).
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        hparams = checkpoint.get("hparams")
        if hparams is None:
            raise ValueError(
                f"Checkpoint {filepath} is missing 'hparams' section."
                " Use save_checkpoint() to save the model."
            )
        model = cls(device=device, **hparams)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, checkpoint
