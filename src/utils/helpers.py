"""
Вспомогательные функции: seed, работа с конфигами, артефакты, логирование.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import torch

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Установка seed для воспроизводимости всех основных источников случайности."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Работа с конфигами
# ---------------------------------------------------------------------------


def _deep_update(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Рекурсивное слияние словарей (override побеждает)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, Mapping)
        ):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(
    path: Union[str, Path, None],
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Загрузка YAML-конфига с опциональными программными переопределениями.

    Args:
        path: Путь к YAML-файлу. Если None - возвращается только overrides.
        overrides: Словарь для рекурсивного переопределения значений.

    Returns:
        Словарь конфигурации.
    """
    config: Dict[str, Any] = {}
    if path is not None:
        if yaml is None:
            raise RuntimeError(
                "Для чтения YAML-конфигов требуется PyYAML (pip install pyyaml)"
            )
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Конфиг не найден: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"YAML-конфиг должен быть словарём: {config_path}")
        config = loaded

    if overrides:
        config = _deep_update(config, overrides)

    return config


def save_config(config: Mapping[str, Any], path: Union[str, Path]) -> None:
    """Сохранение конфигурации в YAML (или JSON при отсутствии PyYAML)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if yaml is None:
        with target.with_suffix(".json").open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False, default=str)
    else:
        with target.open("w", encoding="utf-8") as f:
            yaml.safe_dump(dict(config), f, allow_unicode=True, sort_keys=False)


def get_config_value(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Достаёт значение по точечному ключу (`training.dqn.lr`)."""
    cur: Any = config
    for part in dotted_key.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


# ---------------------------------------------------------------------------
# Run directory и логирование
# ---------------------------------------------------------------------------


def make_run_dir(
    base_dir: Union[str, Path] = "results",
    prefix: str = "run",
    timestamp: bool = True,
) -> Path:
    """
    Создает уникальную директорию для артефактов текущего запуска.

    Структура:
        {base_dir}/{prefix}_{YYYYmmdd_HHMMSS}/
            ├── figures/
            ├── tables/
            ├── models/
            └── logs/
    """
    name = prefix
    if timestamp:
        name = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(base_dir) / name
    for sub in ("figures", "tables", "models", "logs"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def configure_logging(
    run_dir: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    name: str = "rec_sys_edu",
) -> logging.Logger:
    """Настраивает логгер с выводом в stdout и, опционально, в файл внутри run_dir."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # На Windows stdout часто в cp1251 - пробуем переключить на utf-8, чтобы не
    # падать на кириллице и спецсимволах.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if run_dir is not None:
        log_path = Path(run_dir) / "logs" / "run.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Модели
# ---------------------------------------------------------------------------


def save_model(
    model: torch.nn.Module,
    filepath: Union[str, Path],
    metadata: Optional[dict] = None,
) -> None:
    """Сохранение модели с дополнительными метаданными."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_class": model.__class__.__name__,
    }
    if metadata:
        checkpoint.update(metadata)

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    logging.getLogger("rec_sys_edu").info(f"Модель сохранена: {path}")


def load_model(
    model: torch.nn.Module,
    filepath: Union[str, Path],
    device: Optional[torch.device] = None,
) -> dict:
    """Загрузка весов модели из чекпоинта."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logging.getLogger("rec_sys_edu").info(f"Модель загружена: {filepath}")
    return checkpoint


def get_device(force_cpu: bool = False) -> torch.device:
    """Выбирает оптимальное устройство вычислений (CUDA или CPU)."""
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------


def save_metrics(metrics: Mapping[str, Any], path: Union[str, Path]) -> None:
    """Сохраняет словарь метрик в JSON (с корректной сериализацией numpy/torch)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    def _default(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        return str(obj)

    with target.open("w", encoding="utf-8") as f:
        json.dump(dict(metrics), f, indent=2, ensure_ascii=False, default=_default)
