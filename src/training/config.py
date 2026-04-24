"""
Конфигурации для обучения моделей.

Словари-константы (``TRAIN_CONFIG``, ``DEEPFM_CONFIG`` и т.п.) сохранены ради
обратной совместимости с уже написанными скриптами и тестами. Новые точки
входа (``src.api``, ``src.cli``) работают через ``get_default_config(dataset)``
и слияние с YAML-файлами из ``configs/``.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional

# ---------------------------------------------------------------------------
# Базовые словари (back-compat)
# ---------------------------------------------------------------------------

# Конфигурация обучения DQN для ITM-Rec
TRAIN_CONFIG: Dict[str, Any] = {
    "gamma": 0.99,
    "lr": 0.001,
    "tau": 0.01,
    "target_update_freq": 100,
    "batch_size": 64,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
}

# Конфигурация обучения DeepFM+SVD++
DEEPFM_CONFIG: Dict[str, Any] = {
    "embedding_dim": 32,
    "hidden_dims": [128, 64],
    "dropout": 0.2,
    "n_epochs": 50,
    "batch_size": 256,
    "lr": 0.001,
    "weight_decay": 1e-5,
}

# Конфигурация Prioritized Replay Buffer
REPLAY_BUFFER_CONFIG: Dict[str, Any] = {
    "capacity": 10000,
    "alpha": 0.6,
    "beta": 0.4,
    "beta_increment": 0.001,
}

# Конфигурация Dueling DQN архитектуры
DUELING_DQN_CONFIG: Dict[str, Any] = {
    "state_dim": 65,
    "action_dim": 70,
    "hidden_dims": [256, 128, 64],
}


# ---------------------------------------------------------------------------
# Полные дефолтные конфиги для ITM-Rec и OULAD
# ---------------------------------------------------------------------------

ITMREC_DEFAULTS: Dict[str, Any] = {
    "dataset": {
        "type": "itmrec",
        "kaggle_name": "irecsys/itmrec",
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "min_user_ratings": 1,
        "min_item_ratings": 1,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "random_seed": 42,
    },
    "model": {
        "deepfm": copy.deepcopy(DEEPFM_CONFIG),
        "dqn": {
            "state_dim": 65,
            "hidden_dims": [256, 128, 64],
            "dropout": 0.2,
        },
    },
    "training": {
        "dqn": {
            **TRAIN_CONFIG,
            "n_episodes": 200,
            "max_steps_per_episode": 10,
            "eval_interval": 20,
            "use_action_mask": True,
        },
        "replay_buffer": copy.deepcopy(REPLAY_BUFFER_CONFIG),
    },
    "environment": {
        "max_trajectory_length": 10,
        "max_recommendations": 5,
        "novelty_weight": 0.05,
        "novelty_mode": "cosine",
        "reward_mode": "itmrec",
        "reward_weights": {"w1": 0.50, "w2": 0.30, "w3": 0.15, "w4": 0.05},
        "enable_low_reward_termination": False,
        "low_reward_window": 3,
        "low_reward_threshold": 0.2,
    },
    "evaluation": {
        "k": 10,
        "test_users": 20,
        "trajectory_length": 100,
        "n_long_term_users": 30,
        "n_eval_episodes": 10,
        "ablation_modes": ["full", "no_context", "no_demo", "no_history", "no_novelty"],
        "state_ablation_modes": [
            "full_state",
            "no_context",
            "no_demo",
            "no_context_no_demo",
        ],
    },
    "artifacts": {
        "models_dir": "data/models",
        "results_dir": "results",
        "run_prefix": "itmrec",
    },
}

OULAD_DEFAULTS: Dict[str, Any] = {
    "dataset": {
        "type": "oulad",
        "raw_dir": "data/raw/oulad",
        "processed_dir": "data/processed/oulad",
        "kaggle_name": "anlgrbz/student-demographics-online-education-dataoulad",
        "catalog": {
            "mode": "mixed",
            "top_activity_types": 6,
            "top_assessment_types": 3,
            "bucket_delay": True,
        },
        "min_weeks": 2,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "random_seed": 42,
        "proxy_weights": {
            "outcome_mastery": 0.70,
            "outcome_final_result": 0.30,
            "engagement_clicks": 0.35,
            "engagement_active_days": 0.25,
            "engagement_resource_breadth": 0.20,
            "engagement_activity_breadth": 0.10,
            "engagement_regularity": 0.10,
            "selfreg_ontime": 0.40,
            "selfreg_delay": 0.25,
            "selfreg_submission": 0.20,
            "selfreg_continuity": 0.15,
        },
    },
    "model": {
        "deepfm": {
            "embedding_dim": 32,
            "hidden_dims": [128, 64],
            "dropout": 0.2,
            "n_epochs": 30,
            "batch_size": 512,
            "lr": 0.001,
            "weight_decay": 1e-5,
            "heads": ["mastery", "engagement", "selfregulation", "outcome"],
        },
        "dqn": {
            "state_dim": 96,
            "hidden_dims": [256, 128, 64],
            "dropout": 0.2,
        },
    },
    "training": {
        "dqn": {
            "gamma": 0.95,
            "lr": 0.0005,
            "tau": 0.01,
            "target_update_freq": 200,
            "batch_size": 128,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "epsilon_decay": 0.9996,
            "n_episodes": 500,
            "max_steps_per_episode": 40,
            "eval_interval": 50,
            "use_action_mask": True,
        },
        "replay_buffer": {
            "capacity": 50000,
            "alpha": 0.6,
            "beta": 0.4,
            "beta_increment": 0.0005,
        },
    },
    "environment": {
        "max_trajectory_length": 40,
        "max_recommendations": 30,
        "reward_mode": "oulad",
        "use_action_mask": True,
        "novelty_weight": 0.10,
        "terminal_bonus": True,
        "terminal_outcome_weight": 0.5,
        "withdrawn_penalty": 0.5,
        "proxy_decay": 0.6,
        "reward_weights": {
            "outcome": 0.35,
            "mastery": 0.25,
            "engagement": 0.20,
            "selfregulation": 0.20,
        },
        "enable_low_reward_termination": False,
        "low_reward_window": 4,
        "low_reward_threshold": -0.1,
    },
    "evaluation": {
        "k": 10,
        "test_users": 30,
        "trajectory_length": 40,
        "n_long_term_users": 50,
        "n_eval_episodes": 10,
        "ablation_modes": ["full", "no_context", "no_demo", "no_history", "no_novelty"],
        "state_ablation_modes": [
            "full_state",
            "no_context",
            "no_demo",
            "no_context_no_demo",
        ],
    },
    "artifacts": {
        "models_dir": "data/models",
        "results_dir": "results",
        "run_prefix": "oulad",
    },
}


def get_default_config(dataset_type: str) -> Dict[str, Any]:
    """Возвращает дефолтную конфигурацию для заданного датасета."""
    dataset_type = dataset_type.lower()
    if dataset_type == "itmrec":
        return copy.deepcopy(ITMREC_DEFAULTS)
    if dataset_type == "oulad":
        return copy.deepcopy(OULAD_DEFAULTS)
    raise ValueError(f"Неизвестный dataset_type='{dataset_type}'")


def _deep_update(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
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


def merge_with_yaml(
    dataset_type: str,
    yaml_overrides: Optional[Mapping[str, Any]] = None,
    runtime_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Собирает итоговый конфиг: defaults -> yaml -> runtime.

    Args:
        dataset_type: "itmrec" | "oulad".
        yaml_overrides: Содержимое YAML-файла (``load_config()`` уже вернул dict).
        runtime_overrides: Программные переопределения (CLI-флаги, аргументы API).
    """
    config = get_default_config(dataset_type)
    if yaml_overrides:
        config = _deep_update(config, yaml_overrides)
    if runtime_overrides:
        config = _deep_update(config, runtime_overrides)
    return config
