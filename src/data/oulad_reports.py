"""
Создание графиков и отчетных таблиц на основе данных OULAD.

Функции работают с подготовленными данными OULADAnalyzerPreprocessor.
Выделены в отдельный модуль для переиспользования API анализа без загрузки
matplotlib в режиме обучения.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict

import matplotlib

matplotlib.use("Agg")  # Режим без графического интерфейса для CLI и ноутбуков
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .preprocess_oulad import OULADAnalyzerPreprocessor

logger = logging.getLogger("rec_sys_edu")


# ---------------------------------------------------------------------------
# Таблицы
# ---------------------------------------------------------------------------


def save_summary_tables(analyzer: "OULADAnalyzerPreprocessor") -> Dict[str, Any]:
    """Сохраняет сводные таблицы и итоговую статистику."""
    if analyzer.student_features is None:
        raise RuntimeError("Требуется предварительно выполнить build_student_level_features().")

    features = analyzer.student_features.copy()
    daily = analyzer.vle_daily.copy() if analyzer.vle_daily is not None else pd.DataFrame()
    assessments = (
        analyzer.assessment_expected.copy()
        if analyzer.assessment_expected is not None
        else pd.DataFrame()
    )

    summary = {
        "n_student_presentations": int(len(features)),
        "n_unique_students": int(features["id_student"].nunique()),
        "n_unique_modules": int(features["code_module"].nunique()),
        "n_presentations": int(
            features[["code_module", "code_presentation"]].drop_duplicates().shape[0]
        ),
        "n_assessment_rows_after_merge": int(len(assessments)),
        "n_vle_rows_after_merge": int(len(daily)),
        "mean_available_weeks": float(features["available_weeks"].mean()),
        "median_available_weeks": float(features["available_weeks"].median()),
        "mean_mastery": float(features["Mastery"].mean()),
        "mean_engagement": float(features["Engagement"].mean()),
        "mean_selfregulation": float(features["SelfRegulation"].mean()),
        "mean_outcome": float(features["Outcome"].mean()),
    }
    with open(analyzer.tables_dir / "summary_stats.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    result_table = (
        features["final_result"].value_counts(dropna=False)
        .rename_axis("final_result")
        .reset_index(name="count")
    )
    result_table["share"] = result_table["count"] / result_table["count"].sum()
    result_table.to_csv(
        analyzer.tables_dir / "final_result_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    proxy_by_result = (
        features.groupby("final_result", dropna=False)[
            ["Outcome", "Mastery", "Engagement", "SelfRegulation"]
        ]
        .agg(["mean", "median", "std"])
        .round(4)
    )
    proxy_by_result.to_csv(analyzer.tables_dir / "proxy_by_final_result.csv", encoding="utf-8-sig")

    context_cols = [
        "code_module", "code_presentation", "gender",
        "highest_education", "imd_band", "age_band", "disability",
    ]
    ctx_rows = []
    for col in context_cols:
        grouped = features.groupby(col, dropna=False)[
            ["Outcome", "Mastery", "Engagement", "SelfRegulation"]
        ].mean()
        grouped["feature_name"] = col
        grouped["feature_value"] = grouped.index.astype(str)
        ctx_rows.append(grouped.reset_index(drop=True))
    pd.concat(ctx_rows, ignore_index=True).to_csv(
        analyzer.tables_dir / "context_feature_impacts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    corr_cols = [
        "Outcome", "Mastery", "Engagement", "SelfRegulation", "final_result_norm",
        "submission_ratio", "ontime_ratio", "continuity_ratio", "total_clicks", "active_days",
    ]
    corr_cols = [c for c in corr_cols if c in features.columns]
    features[corr_cols].corr(numeric_only=True).to_csv(
        analyzer.tables_dir / "proxy_correlation_matrix.csv", encoding="utf-8-sig"
    )
    return summary


def export_report_tables(analyzer: "OULADAnalyzerPreprocessor") -> None:
    """Экспортирует таблицы для научной работы."""
    if analyzer.student_features is None:
        raise RuntimeError("Требуется предварительно выполнить build_student_level_features().")

    sources = pd.DataFrame([
        {"Источник данных": "courses.csv", "Ключевые поля": "code_module, code_presentation, module_presentation_length",
         "Содержательная роль": "длина module-presentation",
         "Использование далее": "горизонт траектории, замена даты Exam"},
        {"Источник данных": "studentInfo.csv", "Ключевые поля": "id_student, demographics, final_result",
         "Содержательная роль": "демография и итоговый результат",
         "Использование далее": "контекст, profile features, Outcome"},
        {"Источник данных": "studentRegistration.csv", "Ключевые поля": "date_registration, date_unregistration",
         "Содержательная роль": "границы фактического участия",
         "Использование далее": "active_until_day, фильтрация ожидаемых событий"},
        {"Источник данных": "assessments.csv", "Ключевые поля": "assessment_type, date, weight",
         "Содержательная роль": "структура оценивания",
         "Использование далее": "ожидаемые assessment, Mastery, SelfRegulation"},
        {"Источник данных": "studentAssessment.csv", "Ключевые поля": "date_submitted, score, is_banked",
         "Содержательная роль": "результаты оцениваний",
         "Использование далее": "успешность, своевременность, пропуски"},
        {"Источник данных": "vle.csv", "Ключевые поля": "id_site, activity_type, week_from, week_to",
         "Содержательная роль": "тип ресурса и его роль",
         "Использование далее": "широта активности, resource breadth"},
        {"Источник данных": "studentVle.csv", "Ключевые поля": "date, id_site, sum_click",
         "Содержательная роль": "поведенческие логи",
         "Использование далее": "Engagement, weekly trajectories"},
    ])
    sources.to_csv(analyzer.tables_dir / "report_table_oulad_sources.csv",
                   index=False, encoding="utf-8-sig")

    steps = pd.DataFrame([
        {"Шаг": 0, "Преобразование": "Загрузка 7 официальных CSV OULAD",
         "Результат": "Исходные таблицы готовы"},
        {"Шаг": 1, "Преобразование": "Проверка ключей и обязательных колонок",
         "Результат": "Валидные источники для merge"},
        {"Шаг": 2, "Преобразование": "Объединение studentInfo + studentRegistration + courses",
         "Результат": "Базовая единица student-presentation"},
        {"Шаг": 3, "Преобразование": "Построение active_until_day",
         "Результат": "Горизонт траектории"},
        {"Шаг": 4, "Преобразование": "Merge assessments со studentAssessment",
         "Результат": "Ожидаемые/сданные/пропущенные assessment"},
        {"Шаг": 5, "Преобразование": "delay / ontime / weighted_score",
         "Результат": "Признаки успеваемости и самоорганизации"},
        {"Шаг": 6, "Преобразование": "Merge studentVle с vle",
         "Результат": "Обогащённые логи VLE"},
        {"Шаг": 7, "Преобразование": "Агрегация по дням и неделям",
         "Результат": "Признаки вовлечённости и weekly trajectories"},
        {"Шаг": 8, "Преобразование": "Прокси-критерии Outcome/Mastery/Engagement/SelfRegulation",
         "Результат": "Многокритериальная аппроксимация"},
        {"Шаг": 9, "Преобразование": "Сохранение student-level и weekly-level датасетов",
         "Результат": "Готовые таблицы для пайплайна"},
    ])
    steps.to_csv(analyzer.tables_dir / "report_table_oulad_preparation_steps.csv",
                 index=False, encoding="utf-8-sig")

    proxy_table = pd.DataFrame([
        {"Компонент": "Outcome", "Исходные поля": "Mastery, final_result",
         "Логика построения": "0.7·Mastery + 0.3·final_result_norm",
         "Использование": "Интегральный целевой отклик"},
        {"Компонент": "Mastery", "Исходные поля": "score, weight, assessment_type",
         "Логика построения": "Взвешенная нормированная успешность по assessment",
         "Использование": "Прокси академического освоения"},
        {"Компонент": "Engagement", "Исходные поля": "sum_click, active_days, id_site, activity_type",
         "Логика построения": "Интенсивность + широта + регулярность активности",
         "Использование": "Прокси вовлечённости"},
        {"Компонент": "SelfRegulation", "Исходные поля": "date_submitted, date, submission_ratio, continuity_ratio",
         "Логика построения": "Своевременность + доля сдач + устойчивость",
         "Использование": "Прокси самоорганизации"},
    ])
    proxy_table.to_csv(analyzer.tables_dir / "report_table_oulad_proxy_metrics.csv",
                       index=False, encoding="utf-8-sig")

    if analyzer.vle_weekly is not None:
        analyzer.vle_weekly.groupby("week_index")["weekly_clicks"].agg(["mean", "median", "std"]).to_csv(
            analyzer.tables_dir / "weekly_click_statistics.csv", encoding="utf-8-sig"
        )
    if analyzer.assessment_expected is not None:
        analyzer.assessment_expected.groupby("assessment_type")["weight"].agg(
            ["count", "mean", "median", "std"]
        ).to_csv(analyzer.tables_dir / "assessment_type_weight_summary.csv", encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------


def create_all_figures(analyzer: "OULADAnalyzerPreprocessor") -> None:
    """Создает все стандартные диаграммы и графики."""
    if analyzer.student_features is None or analyzer.assessment_expected is None or analyzer.vle_weekly is None:
        raise RuntimeError("Требуется предварительно выполнить подготовку признаков и сводок.")

    _plot_final_result_distribution(analyzer)
    _plot_assessment_structure(analyzer)
    _plot_weekly_vle_dynamics(analyzer)
    _plot_context_impacts(analyzer)
    _plot_timeliness(analyzer)
    _plot_proxy_correlations(analyzer)


def _plot_final_result_distribution(analyzer):
    counts = analyzer.student_features["final_result"].fillna("NA").value_counts().sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.title("Распределение final_result в OULAD")
    plt.xlabel("Финальный результат")
    plt.ylabel("Количество student-presentation")
    plt.tight_layout()
    plt.savefig(analyzer.fig_dir / "oulad_final_result_distribution.png", dpi=200)
    plt.close()


def _plot_assessment_structure(analyzer):
    expected = analyzer.assessment_expected.copy()
    by_type = expected[["id_assessment", "assessment_type", "weight"]].drop_duplicates()
    counts = by_type["assessment_type"].value_counts()
    weights = by_type.groupby("assessment_type")["weight"].mean().sort_values(ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    counts.plot(kind="bar", ax=axes[0])
    axes[0].set_title("Число assessment по типам")
    axes[0].set_xlabel("Тип assessment")
    axes[0].set_ylabel("Количество")
    weights.plot(kind="bar", ax=axes[1])
    axes[1].set_title("Средний вес assessment по типам")
    axes[1].set_xlabel("Тип assessment")
    axes[1].set_ylabel("Средний вес")
    plt.tight_layout()
    plt.savefig(analyzer.fig_dir / "oulad_assessment_structure.png", dpi=200)
    plt.close()


def _plot_weekly_vle_dynamics(analyzer):
    weekly = analyzer.vle_weekly.copy()
    features = analyzer.student_features[["student_presentation_id", "final_result"]].copy()
    weekly = weekly.merge(features, on="student_presentation_id", how="left")
    grouped = (
        weekly.groupby(["week_index", "final_result"], dropna=False)["weekly_clicks"]
        .mean().reset_index()
    )
    pivot = grouped.pivot(index="week_index", columns="final_result", values="weekly_clicks").sort_index()
    plt.figure(figsize=(10, 5.5))
    for col in pivot.columns:
        plt.plot(pivot.index, pivot[col], label=str(col))
    plt.title("Средняя недельная активность в VLE по final_result")
    plt.xlabel("Неделя")
    plt.ylabel("Среднее weekly_clicks")
    plt.legend(title="final_result")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(analyzer.fig_dir / "oulad_weekly_vle_dynamics.png", dpi=200)
    plt.close()


def _plot_context_impacts(analyzer):
    features = analyzer.student_features.copy()
    top = features["code_module"].value_counts().head(8).index.tolist()
    sub = features[features["code_module"].isin(top)]
    grouped = sub.groupby("code_module")[["Outcome", "Mastery", "Engagement", "SelfRegulation"]].mean()
    ax = grouped.plot(kind="bar", figsize=(10, 5.5))
    ax.set_title("Средние прокси-критерии по модулям")
    ax.set_xlabel("Модуль")
    ax.set_ylabel("Среднее значение")
    plt.tight_layout()
    plt.savefig(analyzer.fig_dir / "oulad_context_module_impacts.png", dpi=200)
    plt.close()


def _plot_timeliness(analyzer):
    expected = analyzer.assessment_expected.copy()
    delay = (
        expected[expected["submitted"] == 1]["delay_days_clipped"]
        .fillna(0)
        .to_numpy(dtype=float)
    )
    # Обрезка по оси X: после ~10¹ дней хвост с редкими наблюдениями «съедает» масштаб графика.
    display_max_delay = 10.0
    n_bins = 20

    clipped_mask = delay > display_max_delay
    n_above = int(np.sum(clipped_mask))
    share_above = (n_above / delay.size) if delay.size else 0.0
    delay_vis = delay[~clipped_mask] if delay.size else delay

    bins = np.linspace(0.0, display_max_delay, n_bins + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(delay_vis, bins=bins, color="steelblue", edgecolor="white", linewidth=0.35)
    ax.set_xlim(0.0, display_max_delay)
    ax.set_title("Распределение задержек сдачи assessment")
    if n_above:
        ax.text(
            0.99,
            0.97,
            (
                f"Не показано: {n_above} сдач с задержкой > {display_max_delay:.0f} дней "
                f"({share_above:.1%})."
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            alpha=0.85,
            linespacing=1.15,
        )
    ax.set_xlabel(
        f"Задержка, дней (0 = вовремя; только 0 … {display_max_delay:.0f}; остальной хвост отрезан)"
    )
    ax.set_ylabel("Количество сдач")
    ax.grid(alpha=0.25, which="both", linestyle=":")
    fig.tight_layout()
    fig.savefig(analyzer.fig_dir / "oulad_submission_delay_distribution.png", dpi=200)
    plt.close(fig)


def _plot_proxy_correlations(analyzer):
    corr = analyzer.student_features[["Outcome", "Mastery", "Engagement", "SelfRegulation"]].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.title("Корреляции между прокси-критериями OULAD")
    plt.tight_layout()
    plt.savefig(analyzer.fig_dir / "oulad_proxy_correlation_heatmap.png", dpi=200)
    plt.close()


def run_full_analysis(analyzer: "OULADAnalyzerPreprocessor") -> Dict[str, Any]:
    """Запускает полный анализ: подготовка, таблицы и графики."""
    if analyzer.student_features is None:
        analyzer.run_all()
    summary = save_summary_tables(analyzer)
    export_report_tables(analyzer)
    create_all_figures(analyzer)
    return summary


__all__ = [
    "create_all_figures",
    "export_report_tables",
    "run_full_analysis",
    "save_summary_tables",
]
