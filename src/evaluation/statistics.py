"""
Утилиты статистического анализа для научной работы.

Содержит helper'ы для форматирования результатов t-test/Wilcoxon, доверительных
интервалов и effect size (Cohen's d). Используются тестами H1/H2/H3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: float
    n: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "name": self.name,
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
            "significant": bool(self.significant),
            "effect_size": float(self.effect_size),
            "n": int(self.n),
        }


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Classical Cohen's d для независимых выборок."""
    a_arr = np.asarray(list(a), dtype=float)
    b_arr = np.asarray(list(b), dtype=float)
    if a_arr.size < 2 or b_arr.size < 2:
        return 0.0
    pooled = np.sqrt((a_arr.var(ddof=1) + b_arr.var(ddof=1)) / 2)
    if pooled < 1e-12:
        return 0.0
    return float((a_arr.mean() - b_arr.mean()) / pooled)


def welch_test(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
    *,
    alpha: float = 0.05,
    name: str = "welch_t_test",
) -> TestResult:
    """Welch t-test для независимых выборок (разные дисперсии)."""
    a = np.asarray(list(sample_a), dtype=float)
    b = np.asarray(list(sample_b), dtype=float)
    if a.size < 2 or b.size < 2:
        return TestResult(name, float("nan"), float("nan"), False, 0.0, int(a.size + b.size))
    stat, p = stats.ttest_ind(a, b, equal_var=False)
    return TestResult(
        name=name,
        statistic=float(stat),
        p_value=float(p),
        significant=bool(p < alpha),
        effect_size=cohens_d(a, b),
        n=int(a.size + b.size),
    )


def wilcoxon_test(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
    *,
    alpha: float = 0.05,
    name: str = "wilcoxon",
) -> TestResult:
    """Wilcoxon signed-rank test для парных выборок."""
    a = np.asarray(list(sample_a), dtype=float)
    b = np.asarray(list(sample_b), dtype=float)
    n = min(a.size, b.size)
    if n < 2:
        return TestResult(name, float("nan"), float("nan"), False, 0.0, int(n))
    stat, p = stats.wilcoxon(a[:n], b[:n], zero_method="wilcox", alternative="two-sided")
    return TestResult(
        name=name,
        statistic=float(stat),
        p_value=float(p),
        significant=bool(p < alpha),
        effect_size=cohens_d(a[:n], b[:n]),
        n=int(n),
    )


def bootstrap_ci(
    sample: Sequence[float],
    *,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: Optional[int] = 42,
) -> Dict[str, float]:
    """Бутстреп-доверительный интервал для среднего."""
    arr = np.asarray(list(sample), dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "n": 0}
    rng = np.random.default_rng(random_state)
    means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample_idx = rng.integers(0, arr.size, size=arr.size)
        means[i] = arr[sample_idx].mean()
    lower = float(np.quantile(means, (1.0 - ci) / 2))
    upper = float(np.quantile(means, 1.0 - (1.0 - ci) / 2))
    return {
        "mean": float(arr.mean()),
        "lower": lower,
        "upper": upper,
        "n": int(arr.size),
    }


def compare_against_reference(
    samples: Mapping[str, Sequence[float]],
    reference: str,
    *,
    alpha: float = 0.05,
    test: str = "welch",
) -> Dict[str, Dict[str, float]]:
    """Сравнивает все выборки из ``samples`` с ``reference`` через ``welch``/``wilcoxon``."""
    if reference not in samples:
        raise ValueError(f"Reference {reference!r} не найдён в samples")
    ref = samples[reference]
    report: Dict[str, Dict[str, float]] = {}
    for name, values in samples.items():
        if name == reference:
            continue
        if test == "welch":
            result = welch_test(ref, values, alpha=alpha, name=f"{reference}_vs_{name}")
        elif test == "wilcoxon":
            result = wilcoxon_test(ref, values, alpha=alpha, name=f"{reference}_vs_{name}")
        else:
            raise ValueError(f"Неизвестный тип теста: {test}")
        report[name] = result.to_dict()
    return report
