"""
Публичный Python API прототипа.

Вся логика здесь построена так, чтобы каждую функцию можно было вызвать
как из ноутбука, так и из CLI (``src.cli``). Соглашения:

- Каждая функция принимает ``config`` (dict) и опциональный ``overrides``.
- Каждая функция, которая производит артефакты, создает поддиректорию
  ``results/<prefix>_<timestamp>/`` и пишет в неё figures/tables/models/logs.
- Возвращаемые объекты (``DatasetBundle``, ``model``, ``trainer``, ``metrics``)
  пригодны для дальнейшего использования в ноутбуке.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np
import torch

from .data.loaders import load_dataset as _load_dataset
from .data.schemas import DatasetBundle
from .environment.educational_env import EducationalEnvironment
from .environment.oulad_env import OULADEnvironment
from .evaluation.system_evaluator import run_evaluation_suite
from .evaluation.comparative_tester import ComparativeTester
from .evaluation.long_term_evaluator import LongTermEvaluator
from .evaluation.adaptability import AdaptabilityAnalyzer
from .evaluation.novelty_ablation import NoveltyAblationRunner
from .evaluation.trajectory_visualizer import (
    dump_trajectories_csv,
    plot_reward_trajectories,
    plot_coverage_and_novelty,
)
from .models.deepfm_svdpp import DeepFMSVDPlusPlus
from .models.dueling_dqn import DuelingDQN
from .training.config import merge_with_yaml
from .training.replay_buffer import PrioritizedReplayBuffer
from .training.train_dqn import train_dqn_agent
from .training.train_static import train_static_model
from .training.trainer import DQNTrainer
from .utils.helpers import (
    configure_logging,
    get_device,
    load_config,
    make_run_dir,
    save_config,
    save_metrics,
    set_seed,
)

logger = logging.getLogger("rec_sys_edu")


# ---------------------------------------------------------------------------
# Построение конфига
# ---------------------------------------------------------------------------


def build_config(
    dataset_type: str,
    yaml_path: Union[str, Path, None] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Собирает итоговый конфиг: defaults → yaml → overrides."""
    yaml_cfg = load_config(yaml_path) if yaml_path else {}
    return merge_with_yaml(dataset_type, yaml_cfg, overrides)


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------


def load_dataset_bundle(
    dataset_type: str,
    config: Optional[Mapping[str, Any]] = None,
    yaml_path: Union[str, Path, None] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> DatasetBundle:
    """Возвращает ``DatasetBundle`` для указанного датасета.

    Если ``config`` не передан, он собирается из ``configs/<dataset>.yaml``
    (если путь задан) плюс дефолтов.
    """
    if config is None:
        config = build_config(dataset_type, yaml_path=yaml_path, overrides=overrides)
    return _load_dataset(dataset_type, config)


# ---------------------------------------------------------------------------
# Подготовка рабочей директории
# ---------------------------------------------------------------------------


def prepare_run(
    config: Mapping[str, Any],
    run_name: Optional[str] = None,
    seed: Optional[int] = None,
) -> Path:
    """Создает директорию запуска, настраивает логирование, устанавливает seed и сохраняет конфиг."""
    artifacts = dict(config.get("artifacts", {}))
    base_dir = artifacts.get("results_dir", "results")
    prefix = run_name or artifacts.get("run_prefix", "run")

    run_dir = make_run_dir(base_dir=base_dir, prefix=prefix)
    configure_logging(run_dir)

    if seed is None:
        seed = int(config.get("dataset", {}).get("random_seed", 42))
    set_seed(seed)

    save_config(config, run_dir / "config.yaml")
    logger.info("Запуск: %s", run_dir)
    return run_dir


# ---------------------------------------------------------------------------
# Обучение / загрузка статической модели
# ---------------------------------------------------------------------------


def train_static(
    dataset_type: str,
    config: Optional[Mapping[str, Any]] = None,
    yaml_path: Union[str, Path, None] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    bundle: Optional[DatasetBundle] = None,
    run_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Полный цикл обучения DeepFM+SVD++.

    Returns словарь с ключами ``bundle``, ``model``, ``history``, ``run_dir``.
    """
    if config is None:
        config = build_config(dataset_type, yaml_path=yaml_path, overrides=overrides)

    if run_dir is None:
        run_dir = prepare_run(config, run_name=f"{dataset_type}_deepfm")

    if bundle is None:
        bundle = load_dataset_bundle(dataset_type, config)

    device = device or get_device()
    model, history = train_static_model(bundle, config, run_dir=run_dir, device=device)

    save_metrics({
        "train_losses": history.get("train_losses", []),
        "val_losses": history.get("val_losses", []),
        "best_val_loss": history.get("best_val_loss"),
        "best_checkpoint": history.get("best_checkpoint"),
    }, run_dir / "tables" / "deepfm_history.json")

    return {
        "bundle": bundle,
        "model": model,
        "history": history,
        "run_dir": run_dir,
    }


def load_static_model(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
) -> Tuple[DeepFMSVDPlusPlus, Dict[str, Any]]:
    """Загружает ранее сохранённую DeepFM+SVD++."""
    return DeepFMSVDPlusPlus.load_checkpoint(checkpoint_path, device=device)


# ---------------------------------------------------------------------------
# DQN
# ---------------------------------------------------------------------------


def train_dqn(
    dataset_type: str,
    config: Optional[Mapping[str, Any]] = None,
    yaml_path: Union[str, Path, None] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    bundle: Optional[DatasetBundle] = None,
    deepfm_checkpoint: Union[str, Path, None] = None,
    deepfm_model: Optional[DeepFMSVDPlusPlus] = None,
    run_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Полный цикл обучения Dueling DQN на ``dataset_type``.

    Если ``deepfm_model`` не передан — будет загружен ``deepfm_checkpoint``.
    Возвращает словарь ``{bundle, trainer, history, run_dir, checkpoint}``.
    """
    if config is None:
        config = build_config(dataset_type, yaml_path=yaml_path, overrides=overrides)

    if run_dir is None:
        run_dir = prepare_run(config, run_name=f"{dataset_type}_dqn")

    if bundle is None:
        bundle = load_dataset_bundle(dataset_type, config)

    device = device or get_device()

    if deepfm_model is None:
        if deepfm_checkpoint is None:
            raise ValueError(
                "Нужно передать либо deepfm_model, либо deepfm_checkpoint для train_dqn."
            )
        deepfm_model, _ = load_static_model(deepfm_checkpoint, device=device)

    trainer, history = train_dqn_agent(
        bundle=bundle,
        deepfm_model=deepfm_model,
        config=config,
        run_dir=run_dir,
        device=device,
    )

    save_metrics({
        "training_rewards": history.get("training_rewards", []),
        "training_losses": history.get("training_losses", []),
        "evaluation_scores": history.get("evaluation_scores", []),
        "epsilon_values": history.get("epsilon_values", []),
        "final_evaluation": history.get("final_evaluation", {}),
        "checkpoint": history.get("checkpoint"),
    }, run_dir / "tables" / "dqn_history.json")

    return {
        "bundle": bundle,
        "trainer": trainer,
        "history": history,
        "run_dir": run_dir,
        "checkpoint": history.get("checkpoint"),
    }


def _load_dqn_checkpoint(
    checkpoint_path: Union[str, Path],
    state_dim: int,
    action_dim: int,
    hidden_dims,
    device: torch.device,
) -> DuelingDQN:
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    agent = DuelingDQN(state_dim=state_dim, action_dim=action_dim, hidden_dims=list(hidden_dims), device=device)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    agent.eval()
    return agent


def evaluate_system(
    dataset_type: str,
    hypothesis: str = "all",
    config: Optional[Mapping[str, Any]] = None,
    yaml_path: Union[str, Path, None] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    bundle: Optional[DatasetBundle] = None,
    deepfm_checkpoint: Union[str, Path, None] = None,
    deepfm_model: Optional[DeepFMSVDPlusPlus] = None,
    dqn_checkpoint: Union[str, Path, None] = None,
    dqn_agent: Optional[DuelingDQN] = None,
    run_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Запускает тестовый контур (H1/H2/H3) и сохраняет артефакты."""
    if config is None:
        config = build_config(dataset_type, yaml_path=yaml_path, overrides=overrides)

    if run_dir is None:
        run_dir = prepare_run(config, run_name=f"{dataset_type}_eval")

    if bundle is None:
        bundle = load_dataset_bundle(dataset_type, config)

    device = device or get_device()

    if deepfm_model is None:
        if deepfm_checkpoint is None:
            raise ValueError("Нужно указать deepfm_model или deepfm_checkpoint")
        deepfm_model, _ = load_static_model(deepfm_checkpoint, device=device)

    model_cfg = dict(config.get("model", {}).get("dqn", {})) or {}
    hidden_dims = list(model_cfg.get("hidden_dims", [256, 128, 64]))

    if dqn_agent is None:
        if dqn_checkpoint is None:
            raise ValueError("Нужно указать dqn_agent или dqn_checkpoint")
        dqn_agent = _load_dqn_checkpoint(
            dqn_checkpoint,
            state_dim=int(bundle.state_dim),
            action_dim=int(bundle.n_items),
            hidden_dims=hidden_dims,
            device=device,
        )

    results = run_evaluation_suite(
        bundle=bundle,
        deepfm_model=deepfm_model,
        dqn_agent=dqn_agent,
        config=config,
        run_dir=run_dir,
        hypotheses=hypothesis,
    )
    return {
        "bundle": bundle,
        "results": results,
        "run_dir": run_dir,
    }


# ---------------------------------------------------------------------------
# Построение окружения
# ---------------------------------------------------------------------------


def build_environment(
    bundle: DatasetBundle,
    deepfm_model: DeepFMSVDPlusPlus,
    config: Mapping[str, Any],
) -> Union[EducationalEnvironment, OULADEnvironment]:
    """Создает окружение для обучения RL агента.
    
    Выбирает правильный тип среды в зависимости от типа датасета.
    
    Args:
        bundle: DatasetBundle с загруженными данными
        deepfm_model: Обученная модель DeepFM+SVD++
        config: Конфигурация
        
    Returns:
        EducationalEnvironment для ITM-Rec или OULADEnvironment для OULAD
    """
    dataset_type = bundle.dataset_type.lower()
    if dataset_type == "itmrec":
        dataset_obj = bundle.metadata.get("dataset_object")
        if dataset_obj is None:
            raise RuntimeError("ITM-Rec bundle требует metadata['dataset_object'].")
        return EducationalEnvironment(
            ratings_df=bundle.ratings,
            users_df=bundle.users,
            items_df=bundle.items,
            deepfm_model=deepfm_model,
            dataset=dataset_obj,
            config=config,
        )
    elif dataset_type == "oulad":
        return OULADEnvironment(bundle=bundle, deepfm_model=deepfm_model, config=config)
    else:
        raise NotImplementedError(f"Окружение для '{dataset_type}' не поддерживается")


# ---------------------------------------------------------------------------
# Сравнительное тестирование (гипотеза H2)
# ---------------------------------------------------------------------------


def run_comparative_tests(
    env: Union[EducationalEnvironment, OULADEnvironment],
    deepfm_model: DeepFMSVDPlusPlus,
    dqn_agent: DuelingDQN,
    bundle: DatasetBundle,
    config: Mapping[str, Any],
    run_dir: Optional[Path] = None,
    k: int = 10,
) -> Dict[str, Any]:
    """Запускает сравнительное тестирование моделей.
    
    Сравнивает DQN с базовыми методами по метрикам:
    Precision@K, Recall@K, F1@K, Coverage, Novelty, Diversity.
    
    Args:
        env: Окружение (EducationalEnvironment или OULADEnvironment)
        deepfm_model: Обученная DeepFM+SVD++
        dqn_agent: Обученный DQN агент
        bundle: Dataset bundle
        config: Конфигурация
        run_dir: Директория для сохранения результатов
        k: Количество рекомендаций для оценки
        
    Returns:
        Словарь с результатами сравнения и метриками
    """
    dataset_type = bundle.dataset_type.lower()
    dataset_obj = bundle.metadata.get("dataset_object") if dataset_type == "itmrec" else None
    
    tester = ComparativeTester(
        env=env,
        deepfm_model=deepfm_model,
        dqn_agent=dqn_agent,
        dataset=dataset_obj or bundle,
    )
    
    results = tester.run_comparative_test(
        n_users=min(30, bundle.n_users),
        k=k,
        run_dir=run_dir,
    )
    
    if run_dir:
        save_metrics(results, run_dir / "tables" / "comparative_results.json")
    
    return results


# ---------------------------------------------------------------------------
# Долгосрочная оценка (гипотеза H1)
# ---------------------------------------------------------------------------


def run_long_term_tests(
    env: Union[EducationalEnvironment, OULADEnvironment],
    dqn_agent: DuelingDQN,
    bundle: DatasetBundle,
    config: Mapping[str, Any],
    run_dir: Optional[Path] = None,
    n_users: int = 30,
    trajectory_length: int = 100,
) -> Dict[str, Any]:
    """Запускает долгосрочное тестирование агента.
    
    Вычисляет метрики: CDR, Retention Rate, Learning Slope, Final Coverage.
    
    Args:
        env: Окружение
        dqn_agent: DQN агент
        bundle: Dataset bundle
        config: Конфигурация
        run_dir: Директория для сохранения результатов
        n_users: Количество пользователей для теста
        trajectory_length: Длина траектории для каждого пользователя
        
    Returns:
        Словарь с долгосрочными метриками и визуализациями
    """
    evaluator = LongTermEvaluator(
        env=env,
        agent=dqn_agent,
        bundle=bundle,
        config=config,
    )
    
    results = evaluator.run_long_term_experiment(
        n_users=min(n_users, bundle.n_users),
        trajectory_length=trajectory_length,
        run_dir=run_dir,
    )
    
    if run_dir:
        save_metrics(results, run_dir / "tables" / "long_term_results.json")
        evaluator.visualize_long_term_results(results, run_dir)
    
    return results


# ---------------------------------------------------------------------------
# Адаптивность (гипотеза H2)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Адаптивность (гипотеза H2)
# ---------------------------------------------------------------------------


def run_adaptability_tests(
    env: Union[EducationalEnvironment, OULADEnvironment],
    dqn_agent: DuelingDQN,
    deepfm_model: DeepFMSVDPlusPlus,
    bundle: DatasetBundle,
    config: Mapping[str, Any],
    run_dir: Optional[Path] = None,
    preference_shifts: int = 3,
    trajectory_length: int = 30,
    n_users: int = 20,
) -> Dict[str, Any]:
    """Запускает тесты адаптивности системы.
    
    Проверяет, как система адаптируется к изменениям контекста
    (класс, семестр, блокировка для ITM-Rec; фаза курса для OULAD).
    
    Args:
        env: Окружение
        dqn_agent: DQN агент
        deepfm_model: DeepFM модель
        bundle: Dataset bundle
        config: Конфигурация
        run_dir: Директория для сохранения результатов
        preference_shifts: Количество контекстных сдвигов
        trajectory_length: Длина траектории
        n_users: Количество пользователей
        
    Returns:
        Словарь с метриками адаптивности и стабильности
    """
    analyzer = AdaptabilityAnalyzer(
        env=env,
        agent=dqn_agent,
        deepfm_model=deepfm_model,
        bundle=bundle,
        config=config,
    )
    
    results = analyzer.run_adaptability_test(
        n_users=min(n_users, bundle.n_users),
        trajectory_length=trajectory_length,
        n_shifts=preference_shifts,
        run_dir=run_dir,
    )
    
    if run_dir:
        save_metrics(results, run_dir / "tables" / "adaptability_results.json")
    
    return results


def evaluate_adaptability(
    model: DeepFMSVDPlusPlus,
    env: Union[EducationalEnvironment, OULADEnvironment],
    agent: DuelingDQN,
    bundle: DatasetBundle,
    config: Mapping[str, Any],
    preference_shifts: int = 3,
    trajectory_length: int = 30,
    n_users: Optional[int] = None,
) -> Dict[str, float]:
    """Вычисляет оценку адаптивности: способность системы сохранять стабильность метрик.
    
    AdaptabilityScore = 1 - (σ_P + σ_R) / 2
    
    где:
    - σ_P = стандартное отклонение Precision по контекстным сдвигам
    - σ_R = стандартное отклонение Recall по контекстным сдвигам
    
    Проверяет стабильность метрик при изменении контекста
    (класс, семестр для ITM-Rec; фаза курса для OULAD).
    
    Args:
        model: DeepFM+SVD++ модель
        env: Окружение
        agent: DQN агент
        bundle: Dataset bundle  
        config: Конфигурация
        preference_shifts: Количество контекстных сдвигов для проверки
        trajectory_length: Длина каждой траектории
        n_users: Количество пользователей (если None, используется 20)
        
    Returns:
        Словарь с метриками:
        - adaptability_score: Основная метрика адаптивности (от 0 до 1)
        - precision_std: Стандартное отклонение Precision по сдвигам
        - recall_std: Стандартное отклонение Recall по сдвигам
        - precision_mean: Средняя Precision
        - recall_mean: Средняя Recall
    """
    if n_users is None:
        n_users = min(20, bundle.n_users)
    else:
        n_users = min(n_users, bundle.n_users)
    
    results = run_adaptability_tests(
        env=env,
        dqn_agent=agent,
        deepfm_model=model,
        bundle=bundle,
        config=config,
        preference_shifts=preference_shifts,
        trajectory_length=trajectory_length,
        n_users=n_users,
    )
    
    # Извлекаем метрики из результатов
    # Ищем стабильности Precision и Recall по всем сдвигам
    precision_values = []
    recall_values = []
    
    # Пытаемся собрать данные из структуры results
    if "by_strata" in results:
        for stratum_data in results["by_strata"].values():
            if isinstance(stratum_data, dict):
                if "precision" in stratum_data:
                    precision_values.append(stratum_data["precision"])
                if "recall" in stratum_data:
                    recall_values.append(stratum_data["recall"])
    
    # Если нет стратифицированных данных, используем общие значения
    if not precision_values and "precision" in results:
        precision_values = [results.get("precision", 0.5)]
    if not recall_values and "recall" in results:
        recall_values = [results.get("recall", 0.5)]
    
    # Вычисляем стандартные отклонения
    precision_std = float(np.std(precision_values)) if precision_values else 0.0
    recall_std = float(np.std(recall_values)) if recall_values else 0.0
    
    # Формула оценки адаптивности
    adaptability_score = 1.0 - (precision_std + recall_std) / 2.0
    adaptability_score = float(np.clip(adaptability_score, 0.0, 1.0))
    
    return {
        "adaptability_score": adaptability_score,
        "precision_std": precision_std,
        "recall_std": recall_std,
        "precision_mean": float(np.mean(precision_values)) if precision_values else 0.0,
        "recall_mean": float(np.mean(recall_values)) if recall_values else 0.0,
    }


# ---------------------------------------------------------------------------
# Ablation: новизна (гипотеза H3)
# ---------------------------------------------------------------------------


def run_novelty_ablation(
    env: Union[EducationalEnvironment, OULADEnvironment],
    dqn_agent: DuelingDQN,
    bundle: DatasetBundle,
    config: Mapping[str, Any],
    run_dir: Optional[Path] = None,
    n_users: int = 25,
    trajectory_length: int = 50,
) -> Dict[str, Any]:
    """Запускает ablation-тест новизны.
    
    Сравнивает агента с включённой и отключённой компонентой новизны
    в награде, проверяя баланс между разнообразием и релевантностью.
    
    Args:
        env: Окружение
        dqn_agent: DQN агент
        bundle: Dataset bundle
        config: Конфигурация
        run_dir: Директория для сохранения результатов
        n_users: Количество пользователей
        trajectory_length: Длина траектории
        
    Returns:
        Словарь с результатами ablation-теста
    """
    runner = NoveltyAblationRunner(
        env=env,
        agent=dqn_agent,
        bundle=bundle,
        config=config,
    )
    
    results = runner.run_ablation_study(
        n_users=min(n_users, bundle.n_users),
        trajectory_length=trajectory_length,
        run_dir=run_dir,
    )
    
    if run_dir:
        save_metrics(results, run_dir / "tables" / "novelty_ablation_results.json")
    
    return results


# ---------------------------------------------------------------------------
# Визуализация траекторий
# ---------------------------------------------------------------------------


def visualize_trajectories(
    trajectories: Dict[int, Dict[str, Any]],
    bundle: DatasetBundle,
    run_dir: Optional[Path] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Path]:
    """Создает визуализации траекторий рекомендаций.
    
    Генерирует:
    - CSV-таблицы с траекториями
    - Графики наград за время взаимодействия
    - Графики охвата и новизны
    
    Args:
        trajectories: Словарь траекторий {user_id: trajectory_data}
        bundle: Dataset bundle
        run_dir: Директория для сохранения графиков
        config: Опциональная конфигурация
        
    Returns:
        Словарь с путями сохранённых файлов
    """
    output_paths = {}
    
    if run_dir:
        run_dir = Path(run_dir)
        figures_dir = run_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV-таблицы
        csv_path = dump_trajectories_csv(trajectories, run_dir / "tables")
        output_paths["trajectories_csv"] = csv_path
        
        # Графики наград
        reward_plot = plot_reward_trajectories(trajectories, figures_dir)
        output_paths["reward_trajectories"] = reward_plot
        
        # Графики покрытия и новизны
        coverage_plot = plot_coverage_and_novelty(trajectories, bundle, figures_dir)
        output_paths["coverage_novelty"] = coverage_plot
    
    return output_paths


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------


__all__ = [
    "build_config",
    "load_dataset_bundle",
    "prepare_run",
    "train_static",
    "load_static_model",
    "train_dqn",
    "evaluate_system",
    "build_environment",
    "run_comparative_tests",
    "run_long_term_tests",
    "run_adaptability_tests",
    "evaluate_adaptability",
    "run_novelty_ablation",
    "visualize_trajectories",
]
