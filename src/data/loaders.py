"""
Модуль загрузки данных: ITM-Rec (Kaggle) и OULAD (Kaggle / локально).

Старые функции (``download_kaggle_dataset``, ``load_ratings`` и т.п.)
сохранены ради обратной совместимости с уже написанными скриптами
и ноутбуками.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import pandas as pd

try:  # pragma: no cover — опциональная зависимость
    import kagglehub
except Exception:
    kagglehub = None

logger = logging.getLogger("rec_sys_edu")


# ---------------------------------------------------------------------------
# ITM-Rec loaders (ранее существовавшие функции, без изменений поведения)
# ---------------------------------------------------------------------------


def download_kaggle_dataset(
    dataset_name: str = "irecsys/itmrec",
    output_dir: str = "data/raw",
) -> str:
    """Загрузка датасета с Kaggle через kagglehub."""
    if kagglehub is None:
        raise RuntimeError(
            "Для загрузки датасетов с Kaggle требуется kagglehub (pip install kagglehub)"
        )
    print(f"Загрузка датасета {dataset_name} с Kaggle...")
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = kagglehub.dataset_download(dataset_name)
        print(f"Датасет загружен в: {path}")
        return path
    except Exception as e:
        print(f"Ошибка при загрузке датасета: {e}")
        raise


def _resolve_csv(path: str, filename: str) -> str:
    """Принимает либо путь к файлу, либо к директории; возвращает путь к CSV."""
    if os.path.isfile(path):
        return path
    candidate = os.path.join(path, filename)
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"Файл не найден: {candidate}")
    return candidate


def load_ratings(path: str) -> pd.DataFrame:
    """Загрузка файла ``ratings.csv``."""
    ratings_path = _resolve_csv(path, "ratings.csv")
    ratings = pd.read_csv(ratings_path)
    print(f"Загружено {len(ratings)} записей рейтингов")
    return ratings


def load_users(path: str) -> pd.DataFrame:
    """Загрузка файла ``users.csv``."""
    users_path = _resolve_csv(path, "users.csv")
    users = pd.read_csv(users_path)
    print(f"Загружено {len(users)} пользователей")
    return users


def load_items(path: str) -> pd.DataFrame:
    """Загрузка файла ``items.csv``."""
    items_path = _resolve_csv(path, "items.csv")
    items = pd.read_csv(items_path)
    print(f"Загружено {len(items)} предметов")
    return items


def load_group_ratings(path: str) -> Optional[pd.DataFrame]:
    """Загрузка файла ``group_ratings.csv`` (опциональный)."""
    if os.path.isfile(path):
        group_ratings_path = path
    else:
        group_ratings_path = os.path.join(path, "group_ratings.csv")
    if not os.path.exists(group_ratings_path):
        print("Файл group_ratings.csv не найден, пропускаем")
        return None
    group_ratings = pd.read_csv(group_ratings_path)
    print(f"Загружено {len(group_ratings)} групповых рейтингов")
    return group_ratings


def load_all_data(data_path: str, load_group_ratings: bool = False) -> dict:
    """Загрузка всех таблиц ITM-Rec."""
    print("=" * 50)
    print("ЗАГРУЗКА ДАТАСЕТА ITM-REC")
    print("=" * 50)

    data = {
        "ratings": load_ratings(data_path),
        "users": load_users(data_path),
        "items": load_items(data_path),
    }
    if load_group_ratings:
        data["group_ratings"] = globals()["load_group_ratings"](data_path)

    print("\nРазмеры загруженных таблиц:")
    print(f"- ratings: {data['ratings'].shape}")
    print(f"- users: {data['users'].shape}")
    print(f"- items: {data['items'].shape}")
    if "group_ratings" in data and data["group_ratings"] is not None:
        print(f"- group_ratings: {data['group_ratings'].shape}")
    return data


# ---------------------------------------------------------------------------
# OULAD
# ---------------------------------------------------------------------------


OULAD_REQUIRED_FILES = (
    "courses.csv",
    "assessments.csv",
    "studentInfo.csv",
    "studentRegistration.csv",
    "studentAssessment.csv",
    "vle.csv",
    "studentVle.csv",
)


def download_oulad_dataset(
    dataset_name: str = "anlgrbz/student-demographics-online-education-dataoulad",
    output_dir: str = "data/raw/oulad",
) -> str:
    """Загрузка датасета OULAD с Kaggle.
    
    Загружает датасет в кэш kagglehub, затем копирует файлы в output_dir.
    """
    import shutil
    
    if kagglehub is None:
        raise RuntimeError(
            "Для загрузки OULAD c Kaggle требуется kagglehub (pip install kagglehub)"
        )
    
    print(f"Загрузка датасета {dataset_name} с Kaggle...")
    cache_path = kagglehub.dataset_download(dataset_name)
    print(f"Датасет загружен в кэш: {cache_path}")
    
    # Копируем файлы из кэша в output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Копирование файлов в {output_dir}...")
    
    for file in Path(cache_path).glob("*.csv"):
        dest = Path(output_dir) / file.name
        if not dest.exists():
            shutil.copy2(file, dest)
            print(f"  {file.name}")
        else:
            print(f"  {file.name} (уже существует)")
    
    print(f"Готово. Датасет в: {output_dir}")
    return output_dir


def verify_oulad_files(data_dir: Union[str, Path]) -> Path:
    """Убеждается, что в ``data_dir`` присутствуют все 7 официальных CSV OULAD."""
    data_path = Path(data_dir)
    missing = [f for f in OULAD_REQUIRED_FILES if not (data_path / f).exists()]
    if missing:
        raise FileNotFoundError(
            "В директории OULAD отсутствуют обязательные файлы:\n"
            + "\n".join(str(data_path / m) for m in missing)
        )
    return data_path


# ---------------------------------------------------------------------------
# Единая точка входа load_dataset
# ---------------------------------------------------------------------------


def load_dataset(dataset_type: str, config: Optional[Mapping[str, Any]] = None):
    """Возвращает DatasetBundle для указанного датасета.

    Эта функция - тонкий диспетчер. Реальная логика сборки находится
    в `preprocess_itmrec.build_itmrec_bundle` и
    `preprocess_oulad.build_oulad_bundle`. Импорты - ленивые, чтобы
    модуль `loaders` оставался легким и не требовал всех зависимостей.
    """

    dataset_type = dataset_type.lower()
    cfg: Dict[str, Any] = dict(config) if config else {}

    if dataset_type == "itmrec":
        from .preprocess_itmrec import build_itmrec_bundle

        return build_itmrec_bundle(cfg)

    if dataset_type == "oulad":
        from .preprocess_oulad import build_oulad_bundle

        return build_oulad_bundle(cfg)

    raise ValueError(
        f"Неизвестный dataset_type='{dataset_type}'. Поддерживаются: 'itmrec', 'oulad'."
    )
