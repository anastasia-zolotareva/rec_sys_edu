"""
Подготовка ITM-Rec датасета и сборка единого ``DatasetBundle``.

Функция ``build_itmrec_bundle(config)``:
1. Ищет processed-таблицы, если их нет — загружает raw с Kaggle.
2. Запускает валидацию и опциональную минимальную фильтрацию.
3. Создаёт ``ITMDataset`` (этот же класс используется для обучения DeepFM).
4. Возвращает ``DatasetBundle``, пригодный для среды и тестов.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from .dataset import ITMDataset
from .loaders import (
    download_kaggle_dataset,
    load_all_data,
)
from .preprocessing import validate_data
from .schemas import DatasetBundle

logger = logging.getLogger("rec_sys_edu")


def _load_ratings_users_items(
    processed_dir: str,
    raw_dir: str,
    kaggle_name: str,
) -> Dict[str, pd.DataFrame]:
    """Сначала пытается загрузить processed-таблицы, затем raw/Kaggle."""
    processed_path = Path(processed_dir)
    ratings_csv = processed_path / "ratings_processed.csv"
    users_csv = processed_path / "users_processed.csv"
    items_csv = processed_path / "items_processed.csv"

    if ratings_csv.exists() and users_csv.exists() and items_csv.exists():
        logger.info("Найдены обработанные CSV в %s — загружаем их", processed_path)
        return {
            "ratings": pd.read_csv(ratings_csv),
            "users": pd.read_csv(users_csv),
            "items": pd.read_csv(items_csv),
        }

    logger.info("Processed-данные не найдены, пытаемся загрузить raw")
    if Path(raw_dir).exists() and any(Path(raw_dir).glob("*.csv")):
        return load_all_data(str(raw_dir))

    data_path = download_kaggle_dataset(kaggle_name, raw_dir)
    return load_all_data(data_path)


def _filter_cold_start(
    ratings: pd.DataFrame,
    min_user_ratings: int,
    min_item_ratings: int,
) -> pd.DataFrame:
    """Отсекает пользователей/айтемы с малым числом взаимодействий (iterative)."""
    cur = ratings.copy()
    for _ in range(5):
        user_counts = cur["UserID"].value_counts()
        good_users = user_counts[user_counts >= min_user_ratings].index
        cur = cur[cur["UserID"].isin(good_users)]

        item_counts = cur["Item"].value_counts()
        good_items = item_counts[item_counts >= min_item_ratings].index
        cur = cur[cur["Item"].isin(good_items)]
    return cur


def build_itmrec_bundle(config: Optional[Mapping[str, Any]] = None) -> DatasetBundle:
    """Собирает ``DatasetBundle`` для ITM-Rec.

    Args:
        config: Конфигурация (формат — из ``configs/itmrec.yaml``). Поддерживаются
            ключи ``dataset.*``.
    """
    cfg = dict(config) if config else {}
    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg.get("dataset", {}), Mapping) else {}

    processed_dir = dataset_cfg.get("processed_dir", "data/processed")
    raw_dir = dataset_cfg.get("raw_dir", "data/raw")
    kaggle_name = dataset_cfg.get("kaggle_name", "irecsys/itmrec")
    min_user_ratings = int(dataset_cfg.get("min_user_ratings", 1))
    min_item_ratings = int(dataset_cfg.get("min_item_ratings", 1))

    raw = _load_ratings_users_items(processed_dir, raw_dir, kaggle_name)
    ratings, users, items = raw["ratings"], raw["users"], raw["items"]

    validate_data(ratings, users, items)

    if min_user_ratings > 1 or min_item_ratings > 1:
        before = len(ratings)
        ratings = _filter_cold_start(ratings, min_user_ratings, min_item_ratings)
        logger.info(
            "Cold-start filter: %d -> %d ratings (min_user=%d, min_item=%d)",
            before,
            len(ratings),
            min_user_ratings,
            min_item_ratings,
        )

    dataset = ITMDataset(ratings, users, items)

    item_popularity = (
        dataset.ratings["ItemID_encoded"].value_counts().to_dict()
    )

    state_dim = 32 + 10 + 6 + 15 + 2  # см. docstring EducationalEnvironment

    bundle = DatasetBundle(
        dataset_type="itmrec",
        ratings=dataset.ratings,
        users=dataset.users,
        items=dataset.items,
        target_columns=["Rating", "App", "Data", "Ease"],
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        state_dim=state_dim,
        context_columns=["Class_encoded", "Semester_encoded", "Lockdown_encoded"],
        context_sizes=[dataset.n_classes, dataset.n_semesters, dataset.n_lockdowns],
        encoders={
            "user": dataset.user_encoder,
            "item": dataset.item_encoder,
            "class": dataset.class_encoder,
            "semester": dataset.semester_encoder,
            "lockdown": dataset.lockdown_encoder,
        },
        scalers={"rating": dataset.rating_scaler},
        item_popularity=item_popularity,
        metadata={
            "processed_dir": processed_dir,
            "raw_dir": raw_dir,
            "kaggle_name": kaggle_name,
            "dataset_object": dataset,
        },
    )

    logger.info("ITM-Rec bundle готов: %s", bundle.describe())
    return bundle
