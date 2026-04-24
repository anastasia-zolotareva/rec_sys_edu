"""
Инструменты визуализации траекторий агентов.

Можно использовать в ноутбуках для отрисовки награды / новизны / покрытия во
времени. Сохраняет PNG/CSV в ``run_dir/figures`` и ``run_dir/tables``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger("rec_sys_edu")


def plot_reward_trajectories(
    trajectories: Sequence[Mapping[str, Any]],
    labels: Optional[Sequence[str]] = None,
    title: str = "Кумулятивная награда по шагам",
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Path | None:
    """Рисует кумулятивную награду по шагам для нескольких траекторий."""
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = list(labels) if labels else [f"user_{i}" for i in range(len(trajectories))]
    for traj, label in zip(trajectories, labels):
        rewards = np.asarray(list(traj.get("rewards", [])), dtype=float)
        if rewards.size == 0:
            continue
        cum = np.cumsum(rewards)
        ax.plot(np.arange(1, cum.size + 1), cum, label=label, alpha=0.8)
    ax.set_xlabel("Шаг")
    ax.set_ylabel("Кумулятивная награда")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)

    path = None
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        path = save_path
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_coverage_and_novelty(
    trajectories: Sequence[Mapping[str, Any]],
    n_items: int,
    item_popularity: Mapping[int, float],
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Path | None:
    """Отрисовывает динамику покрытия и среднюю новизну по шагам."""
    if not trajectories:
        return None
    max_len = max((len(t.get("actions", [])) for t in trajectories), default=0)
    if max_len == 0:
        return None

    coverage = np.zeros(max_len)
    novelty = np.zeros(max_len)
    counts = np.zeros(max_len)
    max_pop = max(item_popularity.values()) if item_popularity else 1.0

    for traj in trajectories:
        actions = list(traj.get("actions", []))
        seen = set()
        for i, action in enumerate(actions):
            seen.add(action)
            coverage[i] += len(seen) / n_items
            pop = item_popularity.get(int(action), 0)
            novelty[i] += 1.0 - pop / max(max_pop, 1.0)
            counts[i] += 1

    mask = counts > 0
    coverage[mask] /= counts[mask]
    novelty[mask] /= counts[mask]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(coverage, color="tab:blue")
    axes[0].set_title("Coverage (среднее по пользователям)")
    axes[0].set_xlabel("Шаг")
    axes[0].set_ylabel("Coverage")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(novelty, color="tab:orange")
    axes[1].set_title("Novelty (среднее по пользователям)")
    axes[1].set_xlabel("Шаг")
    axes[1].set_ylabel("Novelty")
    axes[1].grid(True, alpha=0.3)

    path = None
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        path = save_path
    if show:
        plt.show()
    plt.close(fig)
    return path


def dump_trajectories_csv(
    trajectories: Sequence[Mapping[str, Any]],
    save_path: Path,
) -> Path:
    """Сохраняет траектории в CSV (wide: одна строка — один шаг)."""
    rows: List[Dict[str, Any]] = []
    for traj in trajectories:
        user_id = traj.get("user_id")
        for step, (action, reward) in enumerate(zip(traj.get("actions", []), traj.get("rewards", []))):
            rows.append({
                "user_id": user_id,
                "step": step,
                "action": int(action),
                "reward": float(reward),
            })
    df = pd.DataFrame(rows)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    return save_path
