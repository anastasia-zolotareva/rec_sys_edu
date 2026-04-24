"""
Высокоуровневый оркестратор оценки системы (гипотезы H1, H2, H3).

- **H1** (долгосрочная полезность) — :class:`LongTermEvaluator`: CDR,
  Retention Rate, Learning Slope, Final Coverage; сравнение DQN с
  Random / Popularity / DeepFM-SVD++ (Static); статистика Welch/Wilcoxon.
- **H2** (адаптивность) — :class:`AdaptabilityAnalyzer`: ``AdaptabilityScore
  = 1 - (σ_P + σ_R) / 2`` по стратам контекста и демографии; плюс
  state-ablation ``full_state`` / ``no_context`` / ``no_demo`` /
  ``no_context_no_demo`` (см. §6.2).
- **H3** (вклад новизны) — :class:`NoveltyAblationRunner`: ablation
  ``full`` / ``no_novelty`` / ``novelty_popularity`` + парный тест;
  расширенный набор метрик (Precision/Recall/F1@K, Coverage, Diversity,
  Novelty, Mean Reward, CDR).

Функция :func:`run_evaluation_suite` принимает bundle/модели/конфиг и пишет
метрики и графики в ``run_dir``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ..data.schemas import DatasetBundle
from ..environment.educational_env import EducationalEnvironment
from ..environment.oulad_env import OULADEnvironment
from ..models.deepfm_svdpp import DeepFMSVDPlusPlus
from ..models.dueling_dqn import DuelingDQN
from ..utils.helpers import save_metrics
from .adaptability import AdaptabilityAnalyzer
from .comparative_tester import ComparativeTester
from .long_term_evaluator import LongTermEvaluator
from .novelty_ablation import NoveltyAblationRunner
from .statistics import compare_against_reference
from .trajectory_visualizer import (
    dump_trajectories_csv,
    plot_coverage_and_novelty,
    plot_reward_trajectories,
)

logger = logging.getLogger("rec_sys_edu")


def _make_env(bundle: DatasetBundle, deepfm_model: DeepFMSVDPlusPlus, config: Mapping[str, Any]):
    """Создает среду под тип датасета (см. также ``training/train_dqn.py``)."""
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
    if dataset_type == "oulad":
        return OULADEnvironment(bundle=bundle, deepfm_model=deepfm_model, config=config)
    raise NotImplementedError(
        f"Evaluator-среда для '{dataset_type}' пока не реализована."
    )


def _collect_ground_truth(bundle: DatasetBundle, user_ids, threshold: float = 3.0) -> Dict[int, list]:
    """Собирает релевантные предметы для каждого пользователя.

    - ITM-Rec: ``Rating > threshold`` (0..5).
    - OULAD: Outcome/Mastery >= 0.5 (прокси-релевантность).
    """
    ratings = bundle.ratings
    dataset_type = bundle.dataset_type.lower()
    gt: Dict[int, list] = {}
    for uid in user_ids:
        user_ratings = ratings[ratings["UserID_encoded"] == int(uid)]
        if dataset_type == "itmrec" and "Rating" in user_ratings.columns:
            positives = user_ratings[user_ratings["Rating"] > threshold]["ItemID_encoded"].tolist()
        elif dataset_type == "oulad" and "Outcome" in user_ratings.columns:
            positives = user_ratings[user_ratings["Outcome"] >= 0.5]["ItemID_encoded"].tolist()
            if not positives and "Mastery" in user_ratings.columns:
                positives = user_ratings[user_ratings["Mastery"] >= 0.5]["ItemID_encoded"].tolist()
        else:
            positives = user_ratings["ItemID_encoded"].tolist()
        gt[int(uid)] = list(map(int, positives))
    return gt


def _context_strata_columns(bundle: DatasetBundle) -> list:
    """Контекстные колонки из ratings (для ITM-Rec — Class/Semester/Lockdown)."""
    if bundle.dataset_type.lower() == "itmrec":
        candidates = ["Class_encoded", "Semester_encoded", "Lockdown_encoded"]
    else:
        candidates = ["Module_encoded", "Presentation_encoded"]
    return [c for c in candidates if c in bundle.ratings.columns]


def _demographic_strata_columns(bundle: DatasetBundle) -> list:
    """Демографические колонки из users (OULAD: пол/возраст/регион/disability,
    ITM-Rec: Gender/AgeBand/Status_encoded)."""
    if bundle.dataset_type.lower() == "oulad":
        candidates = [
            "Gender_encoded",
            "AgeBand_encoded",
            "Region_encoded",
            "Disability_encoded",
            "HighestEducation_encoded",
        ]
    else:
        candidates = ["Gender_encoded", "AgeBand_encoded", "Status_encoded"]
    return [c for c in candidates if c in bundle.users.columns]


# ----------------------------------------------------------------------
# Визуализация H2
# ----------------------------------------------------------------------


def _plot_h2_segment_analysis(
    by_strata: Mapping[str, Mapping[str, Mapping[str, float]]],
    save_path: Path,
) -> Optional[Path]:
    """Рисует сегментный анализ: Precision/Recall/mean_reward по стратам."""
    if not by_strata:
        return None
    n_strata = len(by_strata)
    fig, axes = plt.subplots(n_strata, 1, figsize=(10, 4 * n_strata), squeeze=False)
    for i, (stratum_col, table) in enumerate(by_strata.items()):
        df = pd.DataFrame(table).T.sort_index()
        ax = axes[i, 0]
        x = np.arange(len(df))
        width = 0.3
        ax.bar(x - width, df["precision"].to_numpy(), width, label="Precision", color="tab:blue", alpha=0.8)
        ax.bar(x, df["recall"].to_numpy(), width, label="Recall", color="tab:orange", alpha=0.8)
        ax.bar(x + width, df["mean_reward"].to_numpy(), width, label="Mean Reward", color="tab:green", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in df.index], rotation=30, ha="right")
        ax.set_title(f"Сегментный анализ по колонке {stratum_col}")
        ax.set_ylabel("Значение метрики")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(loc="best")
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _plot_h2_state_ablation(
    ablation_results: Mapping[str, Mapping[str, float]],
    save_path: Path,
) -> Optional[Path]:
    """Рисует сравнение state-ablation по AdaptabilityScore / stability / P / R / CDR."""
    if not ablation_results:
        return None
    variants = list(ablation_results.keys())
    metrics_layout = [
        ("adaptability_score", "AdaptabilityScore", "tab:blue"),
        ("precision_stability", "PrecisionStability (sigma_P)", "tab:orange"),
        ("recall_stability", "RecallStability (sigma_R)", "tab:green"),
        ("per_user_precision", "Precision@K", "tab:red"),
        ("per_user_recall", "Recall@K", "tab:purple"),
        ("cdr", "Cumulative Discounted Reward (CDR)", "tab:brown"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, title, color) in zip(axes.flatten(), metrics_layout):
        values = [float(ablation_results[v].get(key, 0.0)) for v in variants]
        ax.bar(variants, values, color=color, alpha=0.8)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("H2: Ablation по конфигурациям состояния (state_ablation)", fontsize=13)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ----------------------------------------------------------------------
# H3: расчёт дополнительных метрик и визуализация
# ----------------------------------------------------------------------


def _h3_additional_metrics(
    env,
    agent,
    variants: List[Mapping[str, Any]],
    n_episodes: int,
    max_steps: int,
    ground_truth_by_user: Mapping[int, List[int]],
    k: int,
    gamma: float,
    bundle: Optional[DatasetBundle] = None,
    max_pool_size: int = 500,
) -> Dict[str, Dict[str, float]]:
    """Для каждого варианта среды считает Precision/Recall/F1/Coverage/CDR.

    Чтобы Precision/Recall/F1 не были тождественно нулевыми, в пул эпизодов
    попадают только пользователи с непустым ground truth. При наличии
    ``bundle`` недостающие ground truth собираются на лету через
    :func:`_collect_ground_truth`.
    """
    from .metrics import (
        calculate_coverage,
        calculate_cumulative_discounted_reward,
        calculate_f1_at_k,
        calculate_precision_at_k,
        calculate_recall_at_k,
    )

    results: Dict[str, Dict[str, float]] = {}
    unique_users = env.ratings["UserID_encoded"].unique() if hasattr(env, "ratings") else []
    all_users = list(unique_users)
    # Приоритетно используем уже известные ground truth с непустыми позитивами.
    gt_cache: Dict[int, List[int]] = {
        int(uid): list(map(int, gt))
        for uid, gt in ground_truth_by_user.items()
        if gt
    }
    user_pool = list(gt_cache.keys())
    # Если ground truth почти пуст, подбираем дополнительных пользователей
    # с позитивами из bundle (в пределах ``max_pool_size``).
    if bundle is not None and len(user_pool) < max_pool_size and all_users:
        rng_pool = np.random.default_rng(123)
        candidates = list(all_users)
        rng_pool.shuffle(candidates)
        known = set(gt_cache.keys())
        for uid in candidates:
            uid_int = int(uid)
            if uid_int in known:
                continue
            extra_gt = _collect_ground_truth(bundle, [uid_int]).get(uid_int, [])
            if extra_gt:
                gt_cache[uid_int] = list(map(int, extra_gt))
                user_pool.append(uid_int)
                if len(user_pool) >= max_pool_size:
                    break
    if not user_pool:
        logger.warning(
            "H3 additional metrics: ground truth пуст для всех пользователей, "
            "Precision/Recall/F1 будут нулевыми.",
        )
        user_pool = list(all_users)
    else:
        logger.info(
            "H3 additional metrics: пул пользователей с ground truth = %d (из %d)",
            len(user_pool),
            len(all_users) if all_users else len(user_pool),
        )
    n_items = getattr(env.dataset, "n_items", 1)
    for variant in variants:
        name = variant.get("name", "variant")
        overrides = {key: val for key, val in variant.items() if key != "name"}
        backup = {key: getattr(env, key, None) for key in overrides}
        for key, val in overrides.items():
            setattr(env, key, val)
        try:
            precisions: List[float] = []
            recalls: List[float] = []
            f1s: List[float] = []
            cdrs: List[float] = []
            coverages: List[float] = []
            rng = np.random.default_rng(42)
            for _ in range(n_episodes):
                if user_pool:
                    uid = int(rng.choice(user_pool))
                    state = env.reset(user_id=uid)
                else:
                    state = env.reset()
                    uid = int(getattr(env, "current_user", 0))
                actions: List[int] = []
                rewards: List[float] = []
                for _ in range(max_steps):
                    mask = None
                    if hasattr(env, "get_action_mask"):
                        try:
                            mask = env.get_action_mask()
                        except Exception:
                            mask = None
                    action = int(agent.get_action(state, epsilon=0.01, action_mask=mask))
                    next_state, reward, done, _info = env.step(action)
                    actions.append(action)
                    rewards.append(float(reward))
                    state = next_state
                    if done:
                        break
                gt = gt_cache.get(uid)
                if gt is None:
                    gt = list(ground_truth_by_user.get(uid, []))
                    if not gt and bundle is not None:
                        gt = list(
                            _collect_ground_truth(bundle, [uid]).get(uid, [])
                        )
                    gt_cache[uid] = gt
                precisions.append(calculate_precision_at_k(actions, gt, k, n_items))
                recalls.append(calculate_recall_at_k(actions, gt, k, n_items))
                f1s.append(calculate_f1_at_k(actions, gt, k, n_items))
                coverages.append(calculate_coverage(actions, n_items))
                cdrs.append(calculate_cumulative_discounted_reward(rewards, gamma=gamma))
        finally:
            for key, val in backup.items():
                setattr(env, key, val)
        results[name] = {
            f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
            f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
            f"f1@{k}": float(np.mean(f1s)) if f1s else 0.0,
            "coverage": float(np.mean(coverages)) if coverages else 0.0,
            "cdr": float(np.mean(cdrs)) if cdrs else 0.0,
        }
    return results


def _plot_h3_metrics(
    ablation_serialized: List[Mapping[str, Any]],
    extra_metrics: Mapping[str, Mapping[str, float]],
    save_path: Path,
    k: int = 10,
) -> Optional[Path]:
    """Строит графики всех метрик H3 (Novelty/Coverage/Diversity/P/R/F1/CDR/Mean Reward)."""
    if not ablation_serialized:
        return None
    df = pd.DataFrame(ablation_serialized)
    if "variant" not in df.columns:
        return None
    df = df.set_index("variant")
    variants = df.index.tolist()
    extra_df = pd.DataFrame(extra_metrics).T.reindex(variants)

    metrics_layout = [
        (df["mean_reward"].to_numpy(), "Mean Reward (эпизод)", "tab:blue"),
        (df["mean_novelty"].to_numpy(), "Mean Novelty", "tab:orange"),
        (df["mean_diversity"].to_numpy(), "Mean Diversity", "tab:green"),
        (extra_df.get("coverage", pd.Series([0.0] * len(variants))).to_numpy(), "Coverage", "tab:red"),
        (extra_df.get(f"precision@{k}", pd.Series([0.0] * len(variants))).to_numpy(), f"Precision@{k}", "tab:purple"),
        (extra_df.get(f"recall@{k}", pd.Series([0.0] * len(variants))).to_numpy(), f"Recall@{k}", "tab:brown"),
        (extra_df.get(f"f1@{k}", pd.Series([0.0] * len(variants))).to_numpy(), f"F1@{k}", "tab:pink"),
        (extra_df.get("cdr", pd.Series([0.0] * len(variants))).to_numpy(), "Cumulative Discounted Reward (CDR)", "tab:gray"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, (values, title, color) in zip(axes.flatten(), metrics_layout):
        ax.bar(variants, values, color=color, alpha=0.85)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("H3: сравнение конфигураций награды (ablation новизны)", fontsize=13)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ----------------------------------------------------------------------
# H2: state-ablation runner
# ----------------------------------------------------------------------


def _run_state_ablation_h2(
    env,
    dqn_agent,
    sample_users,
    trajectory_length: int,
    ground_truth: Mapping[int, List[int]],
    strata: List[str],
    users_df: Optional[pd.DataFrame],
    ratings_df: Optional[pd.DataFrame],
    k: int,
    gamma: float,
    state_modes: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Повторяет H2-тест с разными конфигурациями состояния агента."""
    from .metrics import calculate_cumulative_discounted_reward

    ablation: Dict[str, Dict[str, Any]] = {}
    original_mode = getattr(env, "state_ablation", None)
    try:
        for mode in state_modes:
            env.state_ablation = None if mode == "full_state" else mode
            analyzer = AdaptabilityAnalyzer(env, dqn_agent, strata_columns=strata)
            trajectories = analyzer.collect_trajectories(
                sample_users, trajectory_length=trajectory_length
            )
            result = analyzer.analyze(
                trajectories,
                ground_truth,
                users_df=users_df,
                ratings_df=ratings_df,
                k=k,
            )
            cdrs = [
                calculate_cumulative_discounted_reward(t.get("rewards", []), gamma=gamma)
                for t in trajectories
            ]
            payload = result.to_dict()
            payload["cdr"] = float(np.mean(cdrs)) if cdrs else 0.0
            payload["n_users"] = len(trajectories)
            ablation[mode] = payload
            logger.info(
                "H2 state_ablation '%s': adaptability=%.3f, P=%.3f, R=%.3f, CDR=%.3f",
                mode,
                result.adaptability_score,
                result.per_user_precision,
                result.per_user_recall,
                payload["cdr"],
            )
    finally:
        env.state_ablation = original_mode
    return ablation


# ----------------------------------------------------------------------
# Публичный API
# ----------------------------------------------------------------------


def run_evaluation_suite(
    bundle: DatasetBundle,
    deepfm_model: DeepFMSVDPlusPlus,
    dqn_agent: DuelingDQN,
    config: Mapping[str, Any],
    run_dir: Path,
    hypotheses: str = "all",
) -> Dict[str, Any]:
    """Запускает оценку системы под выбранные гипотезы.

    Args:
        bundle: ``DatasetBundle`` с данными.
        deepfm_model: Статическая модель.
        dqn_agent: Обученный DQN.
        config: Сводный конфиг.
        run_dir: Папка для артефактов.
        hypotheses: ``"H1"`` | ``"H2"`` | ``"H3"`` | ``"all"``.

    Returns:
        Словарь с результатами по гипотезам.
    """
    run_dir = Path(run_dir)
    figures_dir = run_dir / "figures"
    tables_dir = run_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    env = _make_env(bundle, deepfm_model, config)
    eval_cfg = dict(config.get("evaluation", {})) or {}
    trajectory_length = int(eval_cfg.get("trajectory_length", 50))
    n_users = int(eval_cfg.get("n_long_term_users", 20))
    n_ablation_episodes = int(eval_cfg.get("n_eval_episodes", 10))
    k = int(eval_cfg.get("k", 10))
    retention_threshold = eval_cfg.get("retention_threshold")
    gamma = float(eval_cfg.get("gamma", 0.99))
    state_ablation_modes = list(
        eval_cfg.get("state_ablation_modes")
        or eval_cfg.get("ablation_modes")
        or ["full_state", "no_context", "no_demo", "no_context_no_demo"]
    )
    state_mode_aliases = {
        "full": "full_state",
        "full_state": "full_state",
        "no_context": "no_context",
        "no_demo": "no_demo",
        "no_context_no_demo": "no_context_no_demo",
    }
    state_ablation_modes = [
        state_mode_aliases.get(mode, mode) for mode in state_ablation_modes
    ]

    unique_users = bundle.ratings["UserID_encoded"].unique()
    rng = np.random.default_rng(int(config.get("dataset", {}).get("random_seed", 42)))
    sample_users = rng.choice(unique_users, size=min(n_users, len(unique_users)), replace=False)
    ground_truth = _collect_ground_truth(bundle, sample_users)

    results: Dict[str, Any] = {"users_sampled": [int(u) for u in sample_users]}

    wanted = {"H1", "H2", "H3"} if hypotheses == "all" else {hypotheses}

    # --- H1: долгосрочная полезность ---
    if "H1" in wanted:
        logger.info("Гипотеза H1: долгосрочная полезность (CDR / Retention / Slope / Coverage)")
        dataset_obj = bundle.metadata.get("dataset_object") or getattr(env, "dataset", None)
        comparative = ComparativeTester(env, deepfm_model, dqn_agent, dataset_obj)
        baseline_funcs: Dict[str, Callable] = {
            "Random": comparative.random_recommender,
            "Popularity": comparative.popularity_recommender,
            "DeepFM-SVD++ (Static)": comparative.static_deepfm_recommender,
        }
        evaluator = LongTermEvaluator(
            env,
            dqn_agent,
            baseline_funcs,
            dataset_type=bundle.dataset_type,
            retention_threshold=retention_threshold,
            k=k,
            gamma=gamma,
            ground_truth_by_user=ground_truth,
        )
        long_term = evaluator.run_long_term_experiment(
            n_users=n_users,
            trajectory_length=trajectory_length,
        )
        serialized: Dict[str, Any] = {}
        for model_name, payload in long_term.items():
            serialized[model_name] = {
                "mean": payload["mean"].to_dict() if hasattr(payload["mean"], "to_dict") else payload["mean"],
                "std": payload["std"].to_dict() if hasattr(payload["std"], "to_dict") else payload["std"],
            }
        significance = evaluator.run_significance_test(long_term, metric="CDR", reference="DQN")
        results["H1"] = {
            "per_model": serialized,
            "significance_cdr": significance,
            "retention_threshold": evaluator.retention_threshold,
            "learning_slope_signal": evaluator.learning_slope_signal,
            "k": k,
        }
        save_metrics(results["H1"], tables_dir / "h1_long_term.json")
        try:
            evaluator.visualize_long_term_results(
                long_term,
                save_path=figures_dir / "h1_long_term_metrics.png",
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("H1 visualization failed: %s", exc)

    # --- H2: адаптивность ---
    if "H2" in wanted:
        logger.info("Гипотеза H2: адаптивность и стабильность по стратам")
        strata = _context_strata_columns(bundle) + _demographic_strata_columns(bundle)
        analyzer = AdaptabilityAnalyzer(env, dqn_agent, strata_columns=strata)
        trajectories = analyzer.collect_trajectories(sample_users, trajectory_length=trajectory_length)
        users_df = bundle.users.copy() if bundle.users is not None else None
        h2 = analyzer.analyze(
            trajectories,
            ground_truth,
            users_df=users_df,
            ratings_df=bundle.ratings,
            k=k,
        )
        h2_payload = h2.to_dict()
        h2_payload["state_ablation_protocol"] = "evaluation_zeroing"

        # State-ablation (full_state / no_context / no_demo / ...)
        state_ablation = _run_state_ablation_h2(
            env,
            dqn_agent,
            sample_users,
            trajectory_length=trajectory_length,
            ground_truth=ground_truth,
            strata=strata,
            users_df=users_df,
            ratings_df=bundle.ratings,
            k=k,
            gamma=gamma,
            state_modes=state_ablation_modes,
        )
        h2_payload["state_ablation"] = state_ablation

        results["H2"] = h2_payload
        save_metrics(results["H2"], tables_dir / "h2_adaptability.json")

        plot_reward_trajectories(
            trajectories[: min(6, len(trajectories))],
            labels=[f"user_{t['user_id']}" for t in trajectories[:6]],
            title="H2: траектории награды (выборка пользователей)",
            save_path=figures_dir / "h2_trajectories.png",
        )
        plot_coverage_and_novelty(
            trajectories,
            n_items=bundle.n_items,
            item_popularity=bundle.item_popularity,
            save_path=figures_dir / "h2_coverage_novelty.png",
        )
        dump_trajectories_csv(trajectories, tables_dir / "h2_trajectories.csv")
        try:
            _plot_h2_segment_analysis(
                h2.by_strata, save_path=figures_dir / "h2_segment_analysis.png"
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("H2 segment plot failed: %s", exc)
        try:
            _plot_h2_state_ablation(
                state_ablation, save_path=figures_dir / "h2_state_ablation.png"
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("H2 state ablation plot failed: %s", exc)

    # --- H3: баланс новизны и релевантности ---
    if "H3" in wanted:
        logger.info("Гипотеза H3: баланс новизны и релевантности")
        variants = [
            {"name": "full"},
            {"name": "no_novelty", "novelty_weight": 0.0},
            {"name": "novelty_popularity", "novelty_mode": "popularity"},
        ]
        ablation = NoveltyAblationRunner(env, dqn_agent).run(
            variants=variants,
            n_episodes=n_ablation_episodes,
            max_steps=trajectory_length,
            user_ids=sample_users,
            seed=int(config.get("dataset", {}).get("random_seed", 42)),
        )
        ablation_serialized = [asdict(r) for r in ablation]
        # Точно такие же варианты пропускаем через дополнительный runner,
        # чтобы посчитать Precision / Recall / F1 / Coverage / CDR.
        h3_extra = _h3_additional_metrics(
            env,
            dqn_agent,
            variants=variants,
            n_episodes=n_ablation_episodes,
            max_steps=trajectory_length,
            ground_truth_by_user=ground_truth,
            k=k,
            gamma=gamma,
            bundle=bundle,
        )
        results["H3"] = {
            "ablation": ablation_serialized,
            "extra_metrics": h3_extra,
        }

        # Парный тест по рядам эпизодных наград (а не по одному среднему).
        samples = {
            r.variant: list(r.episode_rewards) or [r.mean_reward]
            for r in ablation
        }
        if "full" in samples:
            try:
                pairwise = compare_against_reference(samples, reference="full", test="welch")
                results["H3"]["pairwise_vs_full"] = pairwise
            except Exception as exc:  # pragma: no cover
                logger.warning("H3 pairwise test failed: %s", exc)
        save_metrics(results["H3"], tables_dir / "h3_novelty_ablation.json")

        try:
            _plot_h3_metrics(
                ablation_serialized,
                extra_metrics=h3_extra,
                save_path=figures_dir / "h3_metrics.png",
                k=k,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("H3 plot failed: %s", exc)

    save_metrics(results, tables_dir / "evaluation_summary.json")
    return results
