"""
Обучение статической модели DeepFM+SVD++ на ``DatasetBundle``.

Функция ``train_static_model`` работает одинаково для ITM-Rec и OULAD:
она определяет количество выходных голов по ``bundle.target_columns`` и
использует MSE-loss. Для OULAD мы обучаемся на полной ratings-таблице,
где строка = шаг обучения с proxy-критериями.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from ..data.schemas import DatasetBundle
from ..models.deepfm_svdpp import DeepFMSVDPlusPlus
from ..utils.helpers import get_device

logger = logging.getLogger("rec_sys_edu")


class _BundleSupervisedDataset(Dataset):
    """Universal ``Dataset`` working with ``DatasetBundle.ratings``.

    For ITM-Rec ``target_columns`` = [Rating, App, Data, Ease] (scale 1..5,
    normalized to 0..1). For OULAD - already normalized values 0..1.
    """

    def __init__(self, bundle: DatasetBundle, indices: Optional[np.ndarray] = None):
        self.bundle = bundle
        df = bundle.ratings
        self.df = df.iloc[indices].reset_index(drop=True) if indices is not None else df.reset_index(drop=True)

        self.target_columns = list(bundle.target_columns)
        self.context_columns = list(bundle.context_columns)

        # Pre-compute tensors for acceleration
        self._user = self.df["UserID_encoded"].to_numpy(dtype=np.int64)
        self._item = self.df["ItemID_encoded"].to_numpy(dtype=np.int64)
        self._context = [
            self.df[col].to_numpy(dtype=np.int64) if col in self.df.columns else np.zeros(len(self.df), dtype=np.int64)
            for col in self.context_columns
        ]

        # Target normalization
        self._targets = self._prepare_targets()

    def _prepare_targets(self) -> np.ndarray:
        if self.bundle.dataset_type == "itmrec":
            # Rating normalized via rating_scaler - take Rating_norm.
            cols = []
            if "Rating_norm" in self.df.columns:
                cols.append(self.df["Rating_norm"].to_numpy(dtype=np.float32))
            else:
                cols.append((self.df["Rating"].to_numpy(dtype=np.float32) / 5.0))
            for extra in ("App", "Data", "Ease"):
                cols.append(self.df[extra].to_numpy(dtype=np.float32) / 5.0)
            return np.stack(cols, axis=1)

        # OULAD: values already in [0, 1]
        arr = self.df[self.target_columns].to_numpy(dtype=np.float32)
        return np.clip(arr, 0.0, 1.0)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        ctx = {
            "user_id": torch.as_tensor([self._user[idx]], dtype=torch.long),
            "item_id": torch.as_tensor([self._item[idx]], dtype=torch.long),
        }
        default_zeros = torch.as_tensor([0], dtype=torch.long)
        keys = ["class", "semester", "lockdown"]
        for i, key in enumerate(keys):
            if i < len(self._context):
                ctx[key] = torch.as_tensor([self._context[i][idx]], dtype=torch.long)
            else:
                ctx[key] = default_zeros
        return ctx, torch.as_tensor(self._targets[idx], dtype=torch.float32)


def _split_indices(n: int, train_ratio: float, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]


def train_static_model(
    bundle: DatasetBundle,
    config: Mapping[str, Any],
    run_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> Tuple[DeepFMSVDPlusPlus, Dict[str, Any]]:
    """Train DeepFM+SVD++ on a prepared ``DatasetBundle``.

    Returns:
        Tuple ``(model, history)``. ``history`` contains ``train_losses``,
        ``val_losses`` and path to saved best checkpoint.
    """
    device = device or get_device()
    dataset_cfg = dict(config.get("dataset", {})) or {}
    model_cfg = dict(config.get("model", {}).get("deepfm", {})) or {}

    embedding_dim = int(model_cfg.get("embedding_dim", 32))
    hidden_dims = list(model_cfg.get("hidden_dims", [128, 64]))
    dropout_rate = float(model_cfg.get("dropout", 0.2))
    n_epochs = int(model_cfg.get("n_epochs", 30))
    batch_size = int(model_cfg.get("batch_size", 256))
    lr = float(model_cfg.get("lr", 1e-3))
    weight_decay = float(model_cfg.get("weight_decay", 1e-5))

    train_ratio = float(dataset_cfg.get("train_ratio", 0.8))
    val_ratio = float(dataset_cfg.get("val_ratio", 0.1))
    seed = int(dataset_cfg.get("random_seed", 42))

    indices_train, indices_val, indices_test = _split_indices(
        len(bundle.ratings), train_ratio, val_ratio, seed
    )

    train_ds = _BundleSupervisedDataset(bundle, indices_train)
    val_ds = _BundleSupervisedDataset(bundle, indices_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Context sizes (up to three fields).
    csz = list(bundle.context_sizes) + [1, 1, 1]
    n_classes, n_semesters, n_lockdowns = csz[0], csz[1], csz[2]

    model = DeepFMSVDPlusPlus(
        n_users=bundle.n_users,
        n_items=bundle.n_items,
        n_classes=max(1, n_classes),
        n_semesters=max(1, n_semesters),
        n_lockdowns=max(1, n_lockdowns),
        device=device,
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        dataset_type=bundle.dataset_type,
    )

    head_names = list(model.output_heads.keys())

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    history: Dict[str, Any] = {"train_losses": [], "val_losses": []}
    best_val_loss = float("inf")
    best_checkpoint: Optional[Path] = None

    # Folder for checkpoints
    if run_dir is not None:
        checkpoint_dir = Path(run_dir) / "models"
    else:
        checkpoint_dir = Path(config.get("artifacts", {}).get("models_dir", "data/models"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prefix = config.get("artifacts", {}).get("run_prefix", bundle.dataset_type)
    checkpoint_path = checkpoint_dir / f"deepfm_{prefix}_best.pth"

    logger.info(
        "Static model: %s users, %s items, %s heads",
        bundle.n_users, bundle.n_items, len(head_names),
    )

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for features, targets in train_loader:
            for key in features:
                features[key] = features[key].squeeze(-1).to(device)
            targets = targets.to(device)

            preds = model(
                features["user_id"],
                features["item_id"],
                features["class"],
                features["semester"],
                features["lockdown"],
            )

            loss = sum(
                criterion(preds[head_names[i]], targets[:, i])
                for i in range(min(len(head_names), targets.shape[1]))
            ) / max(len(head_names), 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for features, targets in val_loader:
                for key in features:
                    features[key] = features[key].squeeze(-1).to(device)
                targets = targets.to(device)

                preds = model(
                    features["user_id"],
                    features["item_id"],
                    features["class"],
                    features["semester"],
                    features["lockdown"],
                )
                loss = sum(
                    criterion(preds[head_names[i]], targets[:, i])
                    for i in range(min(len(head_names), targets.shape[1]))
                ) / max(len(head_names), 1)
                val_loss += loss.item()
                n_val += 1

        val_loss /= max(n_val, 1)
        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        logger.info(
            "Epoch %d/%d - train_loss=%.4f, val_loss=%.4f",
            epoch + 1, n_epochs, train_loss, val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            extra = {
                "encoders": {k: v for k, v in bundle.encoders.items()},
                "scalers": {k: v for k, v in bundle.scalers.items()},
                "dataset_type": bundle.dataset_type,
                "target_columns": bundle.target_columns,
                "context_columns": bundle.context_columns,
                "context_sizes": bundle.context_sizes,
            }
            model.save_checkpoint(checkpoint_path, extra=extra)
            best_checkpoint = checkpoint_path
            logger.info("Saved best model: %s (val_loss=%.4f)", checkpoint_path, best_val_loss)

    history["best_val_loss"] = best_val_loss
    history["best_checkpoint"] = str(best_checkpoint) if best_checkpoint else None
    history["test_indices"] = indices_test.tolist()
    return model, history
