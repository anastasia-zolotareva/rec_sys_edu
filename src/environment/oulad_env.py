"""
Среда обучения с подкреплением для OULAD-режима.

Ключевые отличия от ITM-Rec среды:
- эпизод = прохождение конкретного student_presentation;
- действие - mixed-step (assessment_type x delay_bucket или activity_type);
- reward считается по инкрементам proxy-критериев Delta-Mastery, Delta-Engagement,
  Delta-SelfRegulation плюс бонус новизны;
- обязательна action_mask, которая учитывает тип шага, неделю и завершенность;
- в state-вектор кодируется прогресс, дедлайны и история mixed-step шагов.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .action_mask import (
    action_availability_features,
    any_action_available,
    build_oulad_action_mask,
)
from .oulad_state import DEFAULT_STATE_DIM, encode_oulad_state
from .reward import (
    DEFAULT_OULAD_WEIGHTS,
    calculate_oulad_step_reward,
    calculate_oulad_terminal_bonus,
)

logger = logging.getLogger("rec_sys_edu")


class OULADEnvironment:
    """Симулятор обучающейся среды для OULAD.

    Параметры конфигурации (config['environment'] либо сам config):
    - max_trajectory_length: ограничение на число шагов (по умолчанию 20);
    - max_recommendations: опциональный верхний порог уникальных действий;
    - reward_weights: словарь весов Delta-M/Delta-E/Delta-SR/Delta-outcome;
    - novelty_weight: вес novelty в итоговой награде (по умолчанию 0.10);
    - terminal_outcome_weight / withdrawn_penalty;
    - proxy_decay: насколько быстро текущий proxy подстраивается к фидбеку;
    - use_action_mask: включить action mask (по умолчанию True).
    """

    def __init__(
        self,
        bundle,  # src.data.schemas.DatasetBundle
        deepfm_model,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.bundle = bundle
        self.model = deepfm_model
        self.device = next(self.model.parameters()).device

        cfg = dict(config or {})
        env_cfg = dict(cfg.get("environment", cfg) or {})
        self.max_trajectory_length = int(env_cfg.get("max_trajectory_length", 20))
        self.max_recommendations = int(env_cfg.get("max_recommendations", 15))
        self.novelty_weight = float(env_cfg.get("novelty_weight", 0.10))
        self.reward_weights = dict(env_cfg.get("reward_weights", DEFAULT_OULAD_WEIGHTS))
        self.terminal_outcome_weight = float(env_cfg.get("terminal_outcome_weight", 0.3))
        self.withdrawn_penalty = float(env_cfg.get("withdrawn_penalty", 0.3))
        self.proxy_decay = float(env_cfg.get("proxy_decay", 0.6))
        self.use_action_mask = bool(env_cfg.get("use_action_mask", True))
        self.low_reward_threshold = float(env_cfg.get("low_reward_threshold", -0.1))
        self.low_reward_window = int(env_cfg.get("low_reward_window", 4))
        self.enable_low_reward_termination = bool(
            env_cfg.get("enable_low_reward_termination", False)
        )
        self.state_dim = int(
            cfg.get("model", {}).get("dqn", {}).get("state_dim", bundle.state_dim or DEFAULT_STATE_DIM)
        )
        self.terminal_bonus_enabled = bool(env_cfg.get("terminal_bonus", True))

        # Индексация.
        self.ratings: pd.DataFrame = bundle.ratings.copy()
        self.users: pd.DataFrame = bundle.users.copy()
        self.items_df: pd.DataFrame = bundle.items.copy()
        self.items_meta: Dict[int, Dict[str, Any]] = dict(bundle.metadata.get("items_meta", {}))
        if not self.items_meta:
            # fallback: строим по bundle.items
            self.items_meta = {
                int(row.item_id): row.to_dict() for _, row in self.items_df.iterrows()
            }

        self.n_items = int(bundle.n_items)
        self.n_users = int(bundle.n_users)
        self.n_modules = max(1, int(bundle.context_sizes[0] if bundle.context_sizes else 1))
        self.n_presentations = max(
            1, int(bundle.context_sizes[1] if len(bundle.context_sizes) > 1 else 1)
        )

        self.user_row_by_enc: Dict[int, Mapping[str, Any]] = {
            int(r.UserID_encoded): r.to_dict() for _, r in self.users.iterrows()
        }
        self.student_features: pd.DataFrame = bundle.metadata.get("student_features", pd.DataFrame())
        self.weekly_features: pd.DataFrame = bundle.metadata.get("weekly", pd.DataFrame())
        self.user_ids: List[int] = sorted(self.ratings["UserID_encoded"].unique().tolist())

        # Совместимость с ExperimentRunner/DQNTrainer (они читают env.dataset.n_items).
        self.dataset = SimpleNamespace(
            n_items=self.n_items,
            n_users=self.n_users,
            ratings=self.ratings,
            user_encoder=bundle.encoders.get("user"),
        )

        # Кэши эмбеддингов.
        self.user_embeddings_cache: Dict[int, np.ndarray] = {}
        self.item_embeddings_cache: Dict[int, np.ndarray] = {}
        self._initialize_caches()

        # Эпизодические переменные.
        self._reset_episode_state()

    # ------------------------------------------------------------------
    # Инициализация
    # ------------------------------------------------------------------

    def _initialize_caches(self) -> None:
        with torch.no_grad():
            item_ids = torch.arange(self.n_items, dtype=torch.long, device=self.device)
            item_embs = self.model.item_emb_fm(item_ids).cpu().numpy()
            for idx, emb in enumerate(item_embs):
                self.item_embeddings_cache[int(idx)] = emb

    def _reset_episode_state(self) -> None:
        self.current_user: Optional[int] = None
        self.current_module_idx: int = 0
        self.current_presentation_idx: int = 0
        self.current_week: int = 0
        self.total_weeks: int = 1
        self.trajectory: List[Dict[str, Any]] = []
        self.recommended_items: set = set()
        self.completed_assessments: set = set()
        self.step_count: int = 0
        self.cumulative_reward: float = 0.0
        self.current_proxy: Dict[str, float] = {
            "mastery": 0.0,
            "engagement": 0.0,
            "selfregulation": 0.0,
            "outcome": 0.0,
        }
        self.prev_proxy: Dict[str, float] = dict(self.current_proxy)
        self.is_withdrawn: bool = False
        self.current_context: Dict[str, int] = {
            "module": 0,
            "presentation": 0,
            "class": 0,
            "semester": 0,
            "lockdown": 0,
        }

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def reset(
        self,
        user_id: Optional[int] = None,
        context: Optional[Mapping[str, int]] = None,
    ) -> np.ndarray:
        """Инициализирует эпизод для конкретного ``student_presentation``."""
        self._reset_episode_state()

        if user_id is None:
            self.current_user = int(np.random.choice(self.user_ids))
        else:
            self.current_user = int(user_id)

        user_row = self.user_row_by_enc.get(self.current_user, {})
        if context is None:
            self.current_module_idx = int(user_row.get("Module_encoded", 0))
            self.current_presentation_idx = int(user_row.get("Presentation_encoded", 0))
        else:
            self.current_module_idx = int(context.get("module", 0))
            self.current_presentation_idx = int(context.get("presentation", 0))

        # Экспортируем словарь контекста (back-compat: совместим с API ITM-Rec среды).
        self.current_context = {
            "module": self.current_module_idx,
            "presentation": self.current_presentation_idx,
            # Back-compat поля для рекомендеров, написанных под ITM-Rec API.
            "class": self.current_module_idx,
            "semester": self.current_presentation_idx,
            "lockdown": 0,
        }

        # Вычисляем total_weeks (доступные недели курса).
        self.total_weeks = self._lookup_total_weeks(self.current_user)
        self.current_week = 0

        # Базовый proxy: берём стартовое значение из student_features (если есть).
        stud = self._lookup_student_row(self.current_user)
        if stud is not None:
            self.current_proxy = {
                "mastery": float(stud.get("Mastery", 0.0) or 0.0) * 0.3,
                "engagement": float(stud.get("Engagement", 0.0) or 0.0) * 0.3,
                "selfregulation": float(stud.get("SelfRegulation", 0.0) or 0.0) * 0.3,
                "outcome": float(stud.get("Outcome", 0.0) or 0.0) * 0.2,
            }
            self.is_withdrawn = str(stud.get("final_result", "")).strip().lower() == "withdrawn"
        self.prev_proxy = dict(self.current_proxy)

        return self._get_state()

    def get_action_mask(self) -> np.ndarray:
        """Маска допустимых действий в текущем состоянии."""
        return build_oulad_action_mask(
            self.items_meta,
            current_week=self.current_week,
            total_weeks=self.total_weeks,
            completed_items=self.completed_assessments,
        )

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Выполняет один mixed-step в среде."""
        if action < 0 or action >= self.n_items:
            raise ValueError(f"Invalid action: {action}")

        self.step_count += 1
        mask = self.get_action_mask()
        if self.use_action_mask and mask[action] <= 0:
            return self._get_state(), -0.5, True, {
                "user_id": self.current_user,
                "item_id": int(action),
                "reason": "invalid_action",
                "cumulative_reward": self.cumulative_reward,
            }

        meta = self.items_meta.get(int(action), {})
        self.recommended_items.add(int(action))
        if str(meta.get("kind", "")).lower() == "assessment":
            self.completed_assessments.add(int(action))

        feedback = self._simulate_feedback(action)
        self.prev_proxy = dict(self.current_proxy)
        self._update_proxy(feedback)
        novelty = self._calculate_novelty(int(action))

        step_reward = calculate_oulad_step_reward(
            self.prev_proxy, self.current_proxy, weights=self.reward_weights
        )
        reward = float(step_reward + self.novelty_weight * novelty)

        self.trajectory.append({
            "item_id": int(action),
            "week": int(self.current_week),
            "kind": meta.get("kind"),
            "assessment_type": meta.get("assessment_type"),
            "activity_type": meta.get("activity_type"),
            "mastery": self.current_proxy["mastery"],
            "engagement": self.current_proxy["engagement"],
            "selfregulation": self.current_proxy["selfregulation"],
            "outcome": self.current_proxy["outcome"],
            "reward": reward,
            "novelty": novelty,
            "delta": {
                k: self.current_proxy[k] - self.prev_proxy[k]
                for k in ("mastery", "engagement", "selfregulation", "outcome")
            },
        })
        self.cumulative_reward += reward

        # Шагаем по неделям (каждое действие продвигает на 1 неделю).
        self.current_week = min(self.current_week + 1, self.total_weeks)

        done, reason = self._check_termination()
        if done and self.terminal_bonus_enabled:
            bonus = calculate_oulad_terminal_bonus(
                self.current_proxy,
                is_withdrawn=self.is_withdrawn,
                outcome_weight=self.terminal_outcome_weight,
                withdrawn_penalty=self.withdrawn_penalty,
            )
            reward += bonus
            self.cumulative_reward += bonus
            self.trajectory[-1]["terminal_bonus"] = bonus

        next_state = self._get_state() if not done else np.zeros(self.state_dim, dtype=np.float32)
        info = {
            "user_id": self.current_user,
            "item_id": int(action),
            "kind": meta.get("kind"),
            "novelty": novelty,
            "cumulative_reward": self.cumulative_reward,
            "step_count": self.step_count,
            "week": self.current_week,
            "done_reason": reason,
        }
        return next_state, reward, bool(done), info

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _lookup_total_weeks(self, user_enc: int) -> int:
        stud = self._lookup_student_row(user_enc)
        if stud is not None and "available_weeks" in stud:
            return max(1, int(stud.get("available_weeks", 1)))
        user_ratings = self.ratings[self.ratings["UserID_encoded"] == user_enc]
        if "week_index" in user_ratings.columns and not user_ratings.empty:
            return max(1, int(user_ratings["week_index"].max()) + 1)
        return 20

    def _lookup_student_row(self, user_enc: int) -> Optional[Mapping[str, Any]]:
        user_row = self.user_row_by_enc.get(user_enc)
        if user_row is None:
            return None
        sp_id = user_row.get("student_presentation_id")
        if sp_id is None or self.student_features.empty:
            return user_row
        match = self.student_features[self.student_features["student_presentation_id"] == sp_id]
        if match.empty:
            return user_row
        row = match.iloc[0].to_dict()
        # Дополняем отсутствующие демографические поля из bundle.users
        for k, v in user_row.items():
            row.setdefault(k, v)
        return row

    def _simulate_feedback(self, action: int) -> Dict[str, float]:
        """Возвращает {mastery, engagement, selfregulation, outcome} в [0, 1].

        Использует либо реальные данные из ratings (медиана), либо предсказание DeepFM.
        """
        user_ratings = self.ratings[
            (self.ratings["UserID_encoded"] == self.current_user)
            & (self.ratings["ItemID_encoded"] == int(action))
        ]
        target_cols = [c for c in ("Mastery", "Engagement", "SelfRegulation", "Outcome") if c in user_ratings.columns]
        if not user_ratings.empty and target_cols:
            feedback = {
                "mastery": float(user_ratings["Mastery"].median()),
                "engagement": float(user_ratings["Engagement"].median()),
                "selfregulation": float(user_ratings["SelfRegulation"].median()),
                "outcome": float(user_ratings["Outcome"].median()),
            }
            return {k: float(np.clip(v, 0.0, 1.0)) for k, v in feedback.items()}

        with torch.no_grad():
            user_tensor = torch.LongTensor([int(self.current_user)]).to(self.device)
            item_tensor = torch.LongTensor([int(action)]).to(self.device)
            class_tensor = torch.LongTensor([int(self.current_module_idx)]).to(self.device)
            semester_tensor = torch.LongTensor([int(self.current_presentation_idx)]).to(self.device)
            lockdown_tensor = torch.zeros(1, dtype=torch.long, device=self.device)
            preds = self.model(
                user_tensor, item_tensor, class_tensor, semester_tensor, lockdown_tensor, None
            )
        feedback = {}
        name_map = {
            "mastery": "mastery",
            "engagement": "engagement",
            "selfregulation": "selfregulation",
            "outcome": "outcome",
        }
        for dst, src in name_map.items():
            if src in preds:
                feedback[dst] = float(preds[src].cpu().item())
            else:
                feedback[dst] = 0.0
        return {k: float(np.clip(v, 0.0, 1.0)) for k, v in feedback.items()}

    def _update_proxy(self, feedback: Mapping[str, float]) -> None:
        alpha = float(np.clip(self.proxy_decay, 0.0, 1.0))
        for key in ("mastery", "engagement", "selfregulation", "outcome"):
            new_val = alpha * float(self.current_proxy.get(key, 0.0)) + (1.0 - alpha) * float(feedback.get(key, 0.0))
            self.current_proxy[key] = float(np.clip(new_val, 0.0, 1.0))

    def _calculate_novelty(self, action: int) -> float:
        emb_a = self.item_embeddings_cache.get(action)
        if emb_a is None:
            return 1.0
        emb_a = np.asarray(emb_a, dtype=np.float64)
        norm_a = np.linalg.norm(emb_a)
        if norm_a < 1e-12:
            return 1.0
        max_sim = 0.0
        for j in self.recommended_items:
            if j == action:
                continue
            emb_b = self.item_embeddings_cache.get(j)
            if emb_b is None:
                continue
            emb_b = np.asarray(emb_b, dtype=np.float64)
            norm_b = np.linalg.norm(emb_b)
            if norm_b < 1e-12:
                continue
            sim = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
            max_sim = max(max_sim, sim)
        if not self.recommended_items:
            # Fallback: популярность предмета в каталоге
            pop = self.bundle.item_popularity.get(int(action), 0)
            max_pop = max(self.bundle.item_popularity.values()) if self.bundle.item_popularity else 1
            return float(np.clip(1.0 - pop / max(max_pop, 1), 0.0, 1.0))
        return float(np.clip(1.0 - max_sim, 0.0, 1.0))

    def _check_termination(self) -> Tuple[bool, str]:
        if self.step_count >= self.max_trajectory_length:
            return True, "max_trajectory_length"
        if len(self.recommended_items) >= self.max_recommendations:
            return True, "max_recommendations"
        if self.current_week >= self.total_weeks:
            return True, "course_completed"
        if self.is_withdrawn:
            return True, "withdrawn"
        mask = self.get_action_mask()
        if not any_action_available(mask):
            return True, "no_actions"
        if (
            self.enable_low_reward_termination
            and len(self.trajectory) >= self.low_reward_window
        ):
            recent = [entry["reward"] for entry in self.trajectory[-self.low_reward_window:]]
            if float(np.mean(recent)) < self.low_reward_threshold:
                return True, "low_reward"
        return False, "none"

    def _get_user_embedding(self, user_enc: int) -> np.ndarray:
        if user_enc in self.user_embeddings_cache:
            return self.user_embeddings_cache[user_enc]
        with torch.no_grad():
            tensor = torch.LongTensor([int(user_enc)]).to(self.device)
            emb = self.model.user_emb_fm(tensor).cpu().numpy().flatten()
        self.user_embeddings_cache[user_enc] = emb
        return emb

    def _compute_state_segments(self) -> Dict[str, Tuple[int, int]]:
        """Границы подсегментов состояния для ablation (см. oulad_state).

        Последовательность сегментов:
        ``user(32) | module | presentation | progress(10) | demo(12) |
        history(16) | time(6) | availability(4) | proxy(4)``.
        """
        module_slot = max(self.n_modules, 8)
        presentation_slot = max(self.n_presentations, 4)
        offsets: Dict[str, Tuple[int, int]] = {}
        cursor = 0
        sizes = [
            ("user", 32),
            ("module", module_slot),
            ("presentation", presentation_slot),
            ("progress", 10),
            ("demo", 12),
            ("history", 16),
            ("time", 6),
            ("availability", 4),
            ("proxy", 4),
        ]
        for name, size in sizes:
            offsets[name] = (cursor, cursor + size)
            cursor += size
        # Стандартный state_dim — 96; если модель использует меньший размер,
        # обрезаем диапазоны по self.state_dim.
        offsets = {
            name: (min(a, self.state_dim), min(b, self.state_dim))
            for name, (a, b) in offsets.items()
        }
        return offsets

    def _apply_state_ablation(self, state: np.ndarray) -> np.ndarray:
        """Зануляет отключённые сегменты состояния (для H2 state-ablation)."""
        mode = getattr(self, "state_ablation", None)
        if not mode or mode == "full":
            return state
        segments = self._compute_state_segments()
        # ``context`` для OULAD = явные course slots + временной прогресс курса.
        # Дополнительно обнуляем ``time`` и ``availability``, потому что они
        # тоже кодируют фазу курса/доступность действий и иначе дают утечку
        # контекста в режиме ``no_context``.
        segments_to_zero: List[str] = []
        if mode == "no_context":
            segments_to_zero = [
                "module",
                "presentation",
                "progress",
                "time",
                "availability",
            ]
        elif mode == "no_demo":
            segments_to_zero = ["demo"]
        elif mode == "no_history":
            segments_to_zero = ["history"]
        elif mode == "no_context_no_demo":
            segments_to_zero = [
                "module",
                "presentation",
                "progress",
                "time",
                "availability",
                "demo",
            ]
        else:
            return state
        state = state.copy()
        for name in segments_to_zero:
            lo, hi = segments[name]
            if hi > lo:
                state[lo:hi] = 0.0
        return state

    def _get_state(self) -> np.ndarray:
        user_emb = self._get_user_embedding(self.current_user)
        mask = self.get_action_mask()
        availability = action_availability_features(mask, self.items_meta)
        user_row = self.user_row_by_enc.get(self.current_user, {})
        state = encode_oulad_state(
            user_embedding=user_emb,
            module_index=self.current_module_idx,
            presentation_index=self.current_presentation_idx,
            n_modules=self.n_modules,
            n_presentations=self.n_presentations,
            user_row=user_row,
            current_week=self.current_week,
            total_weeks=self.total_weeks,
            trajectory=self.trajectory,
            current_proxy=self.current_proxy,
            availability=availability,
            max_trajectory_length=self.max_trajectory_length,
            state_dim=self.state_dim,
        )
        return self._apply_state_ablation(state)


__all__ = ["OULADEnvironment"]
