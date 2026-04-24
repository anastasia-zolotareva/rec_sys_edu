"""
Инварианты проекта (§11 ТЗ).

Проверяют «контрактные» свойства без обращения к реальным датасетам:
* state_dim (ITM-Rec = 65, OULAD = 96) читаются из дефолтных конфигов;
* action-mask OULAD корректно режет уже завершённые assessment и экзамены
  вне финальной фазы, но никогда не возвращает полностью нулевую маску;
* ``calculate_itmrec_reward`` растёт с ростом оценок и шумит адекватно
  на novelty;
* ``calculate_oulad_step_reward`` воспроизводит взвешенную сумму приростов;
* ``calculate_oulad_terminal_bonus`` штрафует withdrawn и поощряет Outcome;
* DeepFM+SVD++ чекпоинт roundtrip (save → load_checkpoint).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.environment.action_mask import (
    any_action_available,
    available_action_count,
    build_oulad_action_mask,
)
from src.environment.oulad_env import OULADEnvironment
from src.environment.reward import (
    calculate_cosine_novelty,
    calculate_itmrec_reward,
    calculate_oulad_step_reward,
    calculate_oulad_terminal_bonus,
)
from src.evaluation.novelty_ablation import NoveltyAblationRunner
from src.evaluation.metrics import calculate_learning_slope
from src.models.deepfm_svdpp import DeepFMSVDPlusPlus
from src.training.config import ITMREC_DEFAULTS, OULAD_DEFAULTS


# ---------------------------------------------------------------------------
# state_dim инварианты
# ---------------------------------------------------------------------------


def test_state_dim_itmrec_default_is_65() -> None:
    assert ITMREC_DEFAULTS["model"]["dqn"]["state_dim"] == 65


def test_state_dim_oulad_default_is_96() -> None:
    assert OULAD_DEFAULTS["model"]["dqn"]["state_dim"] == 96


def test_oulad_default_epsilon_decay_matches_stepwise_schedule() -> None:
    assert OULAD_DEFAULTS["training"]["dqn"]["epsilon_decay"] == pytest.approx(0.9996)


def test_oulad_default_state_ablation_modes_are_publication_ready() -> None:
    assert OULAD_DEFAULTS["evaluation"]["state_ablation_modes"] == [
        "full_state",
        "no_context",
        "no_demo",
        "no_context_no_demo",
    ]


# ---------------------------------------------------------------------------
# Action mask OULAD
# ---------------------------------------------------------------------------


def _make_oulad_items():
    return {
        0: {"kind": "vle", "activity_type": "resource"},
        1: {"kind": "vle", "activity_type": "quiz"},
        2: {"kind": "assessment", "assessment_type": "TMA", "bucket_delay": "on_time"},
        3: {"kind": "assessment", "assessment_type": "Exam", "bucket_delay": "on_time"},
    }


def test_mask_filters_completed_assessments() -> None:
    items = _make_oulad_items()
    mask = build_oulad_action_mask(
        items, current_week=5, total_weeks=10, completed_items=[2]
    )
    assert mask[2] == 0.0
    assert mask[0] == 1.0
    assert any_action_available(mask)


def test_mask_blocks_exam_before_final_phase() -> None:
    items = _make_oulad_items()
    mask = build_oulad_action_mask(items, current_week=2, total_weeks=10)
    assert mask[3] == 0.0  # Exam до финальной фазы заблокирован
    assert mask[2] == 1.0


def test_mask_allows_exam_in_final_phase() -> None:
    items = _make_oulad_items()
    mask = build_oulad_action_mask(items, current_week=9, total_weeks=10)
    assert mask[3] == 1.0


def test_mask_restricts_by_available_kinds() -> None:
    items = _make_oulad_items()
    mask = build_oulad_action_mask(
        items, current_week=3, total_weeks=10, available_kinds={"vle"}
    )
    assert mask[2] == 0.0  # Assessment запрещён
    assert mask[0] == 1.0  # VLE разрешён
    assert available_action_count(mask) == 2


def test_mask_never_all_zero_fallback() -> None:
    """Защитный инвариант: пустая маска схлопывается в «разрешить всё»."""
    items = _make_oulad_items()
    mask = build_oulad_action_mask(
        items,
        current_week=0,
        total_weeks=1,
        completed_items=[0, 1, 2, 3],
        available_kinds={"nonexistent"},
    )
    assert any_action_available(mask)


# ---------------------------------------------------------------------------
# Reward ITM-Rec
# ---------------------------------------------------------------------------


def _demo_vector() -> np.ndarray:
    # gender=1, age=[1,0,0,0], class_encoded=0, married=0
    return np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)


def test_itmrec_reward_monotonic_in_feedback() -> None:
    context = {"class": 0, "semester": 0, "lockdown": 0}
    demo = _demo_vector()
    low = calculate_itmrec_reward(
        {"app": 1.0, "data": 1.0, "ease": 1.0}, context, demo, novelty=0.0
    )
    high = calculate_itmrec_reward(
        {"app": 5.0, "data": 5.0, "ease": 5.0}, context, demo, novelty=0.0
    )
    assert high > low


def test_itmrec_reward_respects_novelty_weight() -> None:
    context = {"class": 0, "semester": 0, "lockdown": 0}
    demo = _demo_vector()
    feedback = {"app": 3.0, "data": 3.0, "ease": 3.0}
    without = calculate_itmrec_reward(feedback, context, demo, novelty=0.0)
    with_nov = calculate_itmrec_reward(
        feedback, context, demo, novelty=1.0, novelty_weight=0.1
    )
    assert with_nov == pytest.approx(without + 0.1, rel=1e-6)


def test_cosine_novelty_full_for_unknown_item() -> None:
    assert calculate_cosine_novelty(
        action=42,
        recommended_items=set(),
        item_embeddings_cache={},
    ) == 1.0


def test_cosine_novelty_low_for_identical_vectors() -> None:
    cache = {0: np.array([1.0, 0.0]), 1: np.array([1.0, 0.0])}
    nov = calculate_cosine_novelty(
        action=0, recommended_items={1}, item_embeddings_cache=cache
    )
    assert nov == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Reward OULAD
# ---------------------------------------------------------------------------


def test_oulad_step_reward_is_weighted_delta() -> None:
    prev = {"outcome": 0.2, "mastery": 0.3, "engagement": 0.4, "selfregulation": 0.5}
    new = {"outcome": 0.3, "mastery": 0.3, "engagement": 0.5, "selfregulation": 0.5}
    reward = calculate_oulad_step_reward(prev, new)
    # 0.35*0.1 + 0.2*0.1 = 0.035 + 0.02
    assert reward == pytest.approx(0.35 * 0.1 + 0.20 * 0.1, rel=1e-6)


def test_oulad_step_reward_respects_negative_delta() -> None:
    prev = {"outcome": 0.5, "mastery": 0.5, "engagement": 0.5, "selfregulation": 0.5}
    new = {"outcome": 0.3, "mastery": 0.5, "engagement": 0.5, "selfregulation": 0.5}
    reward = calculate_oulad_step_reward(prev, new)
    assert reward < 0


def test_oulad_terminal_bonus_penalises_withdrawn() -> None:
    bonus_ok = calculate_oulad_terminal_bonus(
        {"outcome": 0.8}, is_withdrawn=False, outcome_weight=0.5
    )
    bonus_withdrawn = calculate_oulad_terminal_bonus(
        {"outcome": 0.8}, is_withdrawn=True, outcome_weight=0.5, withdrawn_penalty=0.3
    )
    assert bonus_ok == pytest.approx(0.4, rel=1e-6)
    assert bonus_withdrawn == pytest.approx(0.4 - 0.3, rel=1e-6)


def test_learning_slope_auto_uses_cumulative_signal_for_oulad() -> None:
    rewards = [0.5, 0.3, 0.1, 0.0, 0.0]
    reward_slope = calculate_learning_slope(rewards, dataset_type="oulad", signal="reward")
    auto_slope = calculate_learning_slope(rewards, dataset_type="oulad", signal="auto")
    assert reward_slope < 0
    assert auto_slope > 0


def test_learning_slope_auto_keeps_reward_signal_for_itmrec() -> None:
    rewards = [0.5, 0.4, 0.3, 0.2, 0.1]
    auto_slope = calculate_learning_slope(rewards, dataset_type="itmrec", signal="auto")
    reward_slope = calculate_learning_slope(rewards, dataset_type="itmrec", signal="reward")
    assert auto_slope == pytest.approx(reward_slope, rel=1e-6)


def _make_stub_oulad_env_for_ablation() -> OULADEnvironment:
    env = OULADEnvironment.__new__(OULADEnvironment)
    env.n_modules = 7
    env.n_presentations = 4
    env.state_dim = 96
    env.state_ablation = None
    return env


def test_oulad_no_context_ablation_zeros_time_and_availability() -> None:
    env = _make_stub_oulad_env_for_ablation()
    env.state_ablation = "no_context"
    state = np.arange(env.state_dim, dtype=np.float32)
    ablated = env._apply_state_ablation(state)
    segments = env._compute_state_segments()

    for name in ("module", "presentation", "progress", "time", "availability"):
        lo, hi = segments[name]
        assert np.all(ablated[lo:hi] == 0.0), f"{name} segment should be zeroed"

    demo_lo, demo_hi = segments["demo"]
    assert np.array_equal(ablated[demo_lo:demo_hi], state[demo_lo:demo_hi])


def test_oulad_no_demo_preserves_context_but_zeros_demo() -> None:
    env = _make_stub_oulad_env_for_ablation()
    env.state_ablation = "no_demo"
    state = np.arange(env.state_dim, dtype=np.float32)
    ablated = env._apply_state_ablation(state)
    segments = env._compute_state_segments()

    demo_lo, demo_hi = segments["demo"]
    assert np.all(ablated[demo_lo:demo_hi] == 0.0)

    module_lo, module_hi = segments["module"]
    time_lo, time_hi = segments["time"]
    assert np.array_equal(ablated[module_lo:module_hi], state[module_lo:module_hi])
    assert np.array_equal(ablated[time_lo:time_hi], state[time_lo:time_hi])


class _H3DummyEnv:
    def __init__(self) -> None:
        self.reset_trace = []
        self.novelty_weight = 0.1

    def reset(self, user_id=None):
        self.reset_trace.append(user_id)
        return np.zeros(4, dtype=np.float32)

    def get_action_mask(self):
        return np.ones(3, dtype=np.float32)

    def step(self, action):
        return np.zeros(4, dtype=np.float32), 1.0, True, {}

    def _calculate_novelty(self, action):
        return 0.5


class _H3DummyAgent:
    def get_action(self, state, epsilon=0.01, action_mask=None):
        return 0


def test_novelty_ablation_reuses_same_user_pool_across_variants() -> None:
    env = _H3DummyEnv()
    runner = NoveltyAblationRunner(env, _H3DummyAgent())
    runner.run(
        variants=[{"name": "full"}, {"name": "no_novelty", "novelty_weight": 0.0}],
        n_episodes=3,
        max_steps=1,
        user_ids=[10, 20, 30],
        seed=7,
    )

    first_variant = env.reset_trace[:3]
    second_variant = env.reset_trace[3:6]
    assert first_variant == second_variant


def test_novelty_ablation_accepts_numpy_user_pool() -> None:
    env = _H3DummyEnv()
    runner = NoveltyAblationRunner(env, _H3DummyAgent())
    runner.run(
        variants=[{"name": "full"}],
        n_episodes=2,
        max_steps=1,
        user_ids=np.array([11, 22], dtype=np.int64),
        seed=3,
    )
    assert len(env.reset_trace) == 2
    assert set(env.reset_trace) == {11, 22}


# ---------------------------------------------------------------------------
# Checkpoint roundtrip
# ---------------------------------------------------------------------------


def test_deepfm_checkpoint_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = DeepFMSVDPlusPlus(
        n_users=10,
        n_items=12,
        n_classes=3,
        n_semesters=4,
        n_lockdowns=2,
        embedding_dim=16,
        hidden_dims=[32, 16],
        dropout_rate=0.0,
        device=device,
    )
    ckpt_path = tmp_path / "deepfm_test.pth"
    model.save_checkpoint(str(ckpt_path), extra={"n_users": 10, "n_items": 12})

    loaded, meta = DeepFMSVDPlusPlus.load_checkpoint(str(ckpt_path), device=device)
    assert isinstance(loaded, DeepFMSVDPlusPlus)
    assert meta is not None

    model.eval()
    loaded.eval()
    user_ids = torch.arange(5, dtype=torch.long)
    item_ids = torch.arange(5, dtype=torch.long)
    cls_ids = torch.zeros(5, dtype=torch.long)
    sem_ids = torch.zeros(5, dtype=torch.long)
    lock_ids = torch.zeros(5, dtype=torch.long)
    with torch.no_grad():
        out_a = model(user_ids, item_ids, cls_ids, sem_ids, lock_ids)
        out_b = loaded(user_ids, item_ids, cls_ids, sem_ids, lock_ids)
    for key in out_a:
        assert torch.allclose(out_a[key], out_b[key], atol=1e-6), f"mismatch in head '{key}'"
