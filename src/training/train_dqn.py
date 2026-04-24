"""
Высокоуровневая функция обучения Dueling DQN агента.

Используется из :mod:`src.api` и из CLI. Скрипт принимает ``DatasetBundle`` и
обученную модель ``DeepFMSVDPlusPlus`` (или путь к чекпоинту) и запускает
стандартный цикл обучения через :class:`DQNTrainer` + :class:`ExperimentRunner`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from ..data.schemas import DatasetBundle
from ..environment.educational_env import EducationalEnvironment
from ..environment.oulad_env import OULADEnvironment
from ..evaluation.experiment_runner import ExperimentRunner
from ..models.deepfm_svdpp import DeepFMSVDPlusPlus
from ..models.dueling_dqn import DuelingDQN
from ..training.replay_buffer import PrioritizedReplayBuffer
from ..training.trainer import DQNTrainer
from ..utils.helpers import get_device

logger = logging.getLogger("rec_sys_edu")


def _make_env(bundle: DatasetBundle, deepfm_model: DeepFMSVDPlusPlus, config: Mapping[str, Any]):
    """Собирает среду под тип датасета в bundle."""
    dataset_type = bundle.dataset_type.lower()
    if dataset_type == "itmrec":
        dataset_obj = bundle.metadata.get("dataset_object")
        if dataset_obj is None:
            raise RuntimeError(
                "Для ITM-Rec bundle должен содержать metadata['dataset_object'] (ITMDataset)."
            )
        return EducationalEnvironment(
            ratings_df=bundle.ratings,
            users_df=bundle.users,
            items_df=bundle.items,
            deepfm_model=deepfm_model,
            dataset=dataset_obj,
            config=config,
        )
    if dataset_type == "oulad":
        return OULADEnvironment(bundle=bundle, deepfm_model=deepfm_model, config=config)
    raise ValueError(f"Неизвестный dataset_type: {dataset_type}")


def train_dqn_agent(
    bundle: DatasetBundle,
    deepfm_model: DeepFMSVDPlusPlus,
    config: Mapping[str, Any],
    run_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> Tuple[DQNTrainer, Dict[str, Any]]:
    """Complete training cycle of Dueling DQN on a prepared environment.

    Args:
        bundle: ``DatasetBundle`` (ITM-Rec or OULAD).
        deepfm_model: Trained static DeepFM+SVD++ model.
        config: Complete config (``dataset/model/training/environment/artifacts``).
        run_dir: Folder for run artifacts; if ``None``, checkpoints go to
            ``config['artifacts']['models_dir']``.
        device: torch.device.

    Returns:
        Tuple ``(trainer, history)``. ``history`` contains training curves and
        final evaluation metrics.
    """
    device = device or get_device()

    model_cfg = dict(config.get("model", {}).get("dqn", {})) or {}
    train_cfg = dict(config.get("training", {}).get("dqn", {})) or {}
    buf_cfg = dict(config.get("training", {}).get("replay_buffer", {})) or {}
    eval_cfg = dict(config.get("evaluation", {})) or {}

    state_dim = int(bundle.state_dim)
    action_dim = int(bundle.n_items)

    hidden_dims = list(model_cfg.get("hidden_dims", [256, 128, 64]))

    deepfm_model.eval()
    deepfm_model.to(device)

    env = _make_env(bundle, deepfm_model, config)

    agent = DuelingDQN(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        device=device,
    )

    buffer = PrioritizedReplayBuffer(
        capacity=int(buf_cfg.get("capacity", 10000)),
        alpha=float(buf_cfg.get("alpha", 0.6)),
        beta=float(buf_cfg.get("beta", 0.4)),
        beta_increment=float(buf_cfg.get("beta_increment", 0.001)),
    )

    trainer_cfg = {
        "gamma": float(train_cfg.get("gamma", 0.99)),
        "lr": float(train_cfg.get("lr", 1e-3)),
        "tau": float(train_cfg.get("tau", 0.01)),
        "target_update_freq": int(train_cfg.get("target_update_freq", 100)),
        "batch_size": int(train_cfg.get("batch_size", 64)),
        "epsilon_start": float(train_cfg.get("epsilon_start", 1.0)),
        "epsilon_end": float(train_cfg.get("epsilon_end", 0.01)),
        "epsilon_decay": float(train_cfg.get("epsilon_decay", 0.995)),
        "use_action_mask": bool(train_cfg.get("use_action_mask", True)),
        "max_steps_per_episode": int(train_cfg.get("max_steps_per_episode", 100)),
    }

    trainer = DQNTrainer(env, agent, buffer, trainer_cfg)

    n_episodes = int(train_cfg.get("n_episodes", 200))
    eval_interval = int(train_cfg.get("eval_interval", 20))
    max_steps_per_episode = int(train_cfg.get("max_steps_per_episode", 100))

    runner = ExperimentRunner(trainer, env)
    training_results = runner.run_training_experiment(
        n_episodes=n_episodes,
        eval_interval=eval_interval,
        max_steps_per_episode=max_steps_per_episode,
    )

    # Final evaluation
    final_eval = trainer.evaluate(n_episodes=int(eval_cfg.get("n_eval_episodes", 10)))
    training_results["final_evaluation"] = final_eval

    # Save checkpoint
    if run_dir is not None:
        checkpoint_dir = Path(run_dir) / "models"
    else:
        checkpoint_dir = Path(config.get("artifacts", {}).get("models_dir", "data/models"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prefix = config.get("artifacts", {}).get("run_prefix", bundle.dataset_type)
    checkpoint_path = checkpoint_dir / f"dqn_{prefix}_checkpoint.pth"
    trainer.save_checkpoint(str(checkpoint_path))
    training_results["checkpoint"] = str(checkpoint_path)

    logger.info(
        "DQN training completed: episodes=%d, final epsilon=%.3f, mean_reward=%.3f",
        n_episodes, trainer.epsilon, final_eval.get("mean_reward", float("nan")),
    )

    return trainer, training_results
