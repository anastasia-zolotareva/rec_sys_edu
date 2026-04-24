"""
Ablation-анализ новизны: сравнение variants с/без novelty bonus и с разными
режимами (cosine / popularity), а также отключением отдельных компонент
состояния (no_context / no_demo / no_history).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

logger = logging.getLogger("rec_sys_edu")


@dataclass
class AblationResult:
    variant: str
    mean_reward: float
    std_reward: float
    mean_novelty: float
    mean_diversity: float
    n_episodes: int
    episode_rewards: List[float] = field(default_factory=list)


class NoveltyAblationRunner:
    """Запускает эпизоды с модифицированными настройками среды."""

    def __init__(self, env, agent) -> None:
        self.env = env
        self.agent = agent

    def _run_episodes(
        self,
        n_episodes: int,
        max_steps: int,
        user_ids: Optional[Sequence[int]] = None,
        seed: int = 42,
    ) -> Dict[str, float]:
        rewards: List[float] = []
        novelties: List[float] = []
        diversities: List[float] = []
        user_pool = [int(uid) for uid in user_ids] if user_ids is not None else []
        if user_pool:
            rng = np.random.default_rng(seed)
            rng.shuffle(user_pool)
        for episode_idx in range(n_episodes):
            if user_pool:
                uid = user_pool[episode_idx % len(user_pool)]
                state = self.env.reset(user_id=uid)
            else:
                uid = None
                state = self.env.reset()
            ep_reward = 0.0
            actions: List[int] = []
            novelty_trace: List[float] = []
            for _ in range(max_steps):
                mask = None
                if hasattr(self.env, "get_action_mask"):
                    try:
                        mask = self.env.get_action_mask()
                    except Exception:
                        mask = None
                action = self.agent.get_action(state, epsilon=0.01, action_mask=mask)
                next_state, reward, done, info = self.env.step(action)
                ep_reward += reward
                actions.append(int(action))
                # Новизна, если среда её считает
                novelty_trace.append(float(self.env._calculate_novelty(action)) if hasattr(self.env, "_calculate_novelty") else 0.0)
                state = next_state
                if done:
                    break
            rewards.append(ep_reward)
            diversities.append(len(set(actions)) / max(len(actions), 1))
            novelties.append(float(np.mean(novelty_trace)) if novelty_trace else 0.0)
        return {
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "std_reward": float(np.std(rewards)) if rewards else 0.0,
            "mean_novelty": float(np.mean(novelties)) if novelties else 0.0,
            "mean_diversity": float(np.mean(diversities)) if diversities else 0.0,
            "episode_rewards": list(rewards),
        }

    def run(
        self,
        variants: Optional[Sequence[Mapping[str, Any]]] = None,
        n_episodes: int = 10,
        max_steps: int = 20,
        user_ids: Optional[Sequence[int]] = None,
        seed: int = 42,
    ) -> List[AblationResult]:
        """Запускает несколько вариантов среды.

        ``variants`` — список словарей с полями ``name`` и любыми атрибутами
        среды, которые нужно переопределить (``novelty_weight``,
        ``novelty_mode``). Оригинальные значения восстанавливаются после
        прогона каждого варианта.
        """
        if variants is None:
            variants = [
                {"name": "full"},
                {"name": "no_novelty", "novelty_weight": 0.0},
                {"name": "novelty_popularity", "novelty_mode": "popularity"},
            ]

        results: List[AblationResult] = []
        for variant in variants:
            name = variant.get("name", "variant")
            overrides = {k: v for k, v in variant.items() if k != "name"}
            backup = {k: getattr(self.env, k, None) for k in overrides}
            for k, v in overrides.items():
                setattr(self.env, k, v)
            try:
                metrics = self._run_episodes(
                    n_episodes,
                    max_steps,
                    user_ids=user_ids,
                    seed=seed,
                )
            finally:
                for k, v in backup.items():
                    setattr(self.env, k, v)
            results.append(AblationResult(
                variant=name,
                mean_reward=metrics["mean_reward"],
                std_reward=metrics["std_reward"],
                mean_novelty=metrics["mean_novelty"],
                mean_diversity=metrics["mean_diversity"],
                n_episodes=n_episodes,
                episode_rewards=metrics.get("episode_rewards", []),
            ))
            logger.info(
                "Ablation variant '%s': mean_reward=%.3f novelty=%.3f diversity=%.3f",
                name, metrics["mean_reward"], metrics["mean_novelty"], metrics["mean_diversity"],
            )
        return results
