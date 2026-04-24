"""
Препроцессинг OULAD: загружает 7 официальных таблиц, вычисляет прокси-критерии
Outcome/Mastery/Engagement/SelfRegulation и формирует DatasetBundle с
mixed-step каталогом рекомендаций для последующего обучения с подкреплением.

Полная логика построения признаков находится в OULADAnalyzerPreprocessor.
Модуль совместим с корневым скриптом oulad_analysis_preparation.py.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None

from .schemas import DatasetBundle

logger = logging.getLogger("rec_sys_edu")


# ---------------------------------------------------------------------------
# Константы и схема
# ---------------------------------------------------------------------------

REQUIRED_FILES = {
    "courses": "courses.csv",
    "assessments": "assessments.csv",
    "studentInfo": "studentInfo.csv",
    "studentRegistration": "studentRegistration.csv",
    "studentAssessment": "studentAssessment.csv",
    "vle": "vle.csv",
    "studentVle": "studentVle.csv",
}

REQUIRED_COLUMNS = {
    "courses": ["code_module", "code_presentation", "module_presentation_length"],
    "assessments": [
        "code_module", "code_presentation", "id_assessment", "assessment_type", "date", "weight",
    ],
    "studentInfo": [
        "code_module", "code_presentation", "id_student", "gender", "region",
        "highest_education", "imd_band", "age_band", "num_of_prev_attempts",
        "studied_credits", "disability", "final_result",
    ],
    "studentRegistration": [
        "code_module", "code_presentation", "id_student",
        "date_registration", "date_unregistration",
    ],
    "studentAssessment": [
        "id_assessment", "id_student", "date_submitted", "is_banked", "score",
    ],
    "vle": [
        "id_site", "code_module", "code_presentation", "activity_type", "week_from", "week_to",
    ],
    "studentVle": [
        "code_module", "code_presentation", "id_student", "id_site", "date", "sum_click",
    ],
}

FINAL_RESULT_MAP = {
    "distinction": 1.0,
    "pass": 0.8,
    "fail": 0.35,
    "withdrawn": 0.0,
    "withdrawal": 0.0,
}


@dataclass
class ProxyWeights:
    outcome_mastery: float = 0.70
    outcome_final_result: float = 0.30
    engagement_clicks: float = 0.35
    engagement_active_days: float = 0.25
    engagement_resource_breadth: float = 0.20
    engagement_activity_breadth: float = 0.10
    engagement_regularity: float = 0.10
    selfreg_ontime: float = 0.40
    selfreg_delay: float = 0.25
    selfreg_submission: float = 0.20
    selfreg_continuity: float = 0.15

    @classmethod
    def from_mapping(cls, data: Mapping[str, float]) -> "ProxyWeights":
        fields = {f.name: getattr(cls(), f.name) for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        for key, value in data.items():
            if key in fields:
                fields[key] = float(value)
        return cls(**fields)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def minmax_series(series: pd.Series) -> pd.Series:
    x = series.astype(float)
    valid = x.dropna()
    if valid.empty:
        return pd.Series(np.zeros(len(x)), index=x.index)
    x_min = float(valid.min())
    x_max = float(valid.max())
    if math.isclose(x_max, x_min):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - x_min) / (x_max - x_min)


def robust_inverse_minmax(series: pd.Series) -> pd.Series:
    fill = series.max() if series.notna().any() else 0.0
    return 1.0 - minmax_series(series.fillna(fill))


def combine_keys(df: pd.DataFrame, keys: List[str], new_col: str) -> pd.DataFrame:
    df = df.copy()
    df[new_col] = df[keys].astype(str).agg("__".join, axis=1)
    return df


# ---------------------------------------------------------------------------
# OULADAnalyzerPreprocessor — полный EDA/feature pipeline
# ---------------------------------------------------------------------------


class OULADAnalyzerPreprocessor:
    """Полный конвейер подготовки данных OULAD."""

    def __init__(
        self,
        data_dir: str | Path,
        output_dir: str | Path,
        weights: ProxyWeights | None = None,
        min_active_days_for_week: int = 1,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.fig_dir = self.output_dir / "figures"
        self.tables_dir = self.output_dir / "tables"
        self.dataset_dir = self.output_dir / "prepared_datasets"
        self.weights = weights or ProxyWeights()
        self.min_active_days_for_week = min_active_days_for_week

        for p in [self.output_dir, self.fig_dir, self.tables_dir, self.dataset_dir]:
            p.mkdir(parents=True, exist_ok=True)

        self.tables: Dict[str, pd.DataFrame] = {}
        self.student_base: pd.DataFrame | None = None
        self.assessment_expected: pd.DataFrame | None = None
        self.assessment_features: pd.DataFrame | None = None
        self.vle_daily: pd.DataFrame | None = None
        self.vle_weekly: pd.DataFrame | None = None
        self.student_features: pd.DataFrame | None = None

    def load_tables(self) -> Dict[str, pd.DataFrame]:
        missing: List[str] = []
        for key, filename in REQUIRED_FILES.items():
            filepath = self.data_dir / filename
            if not filepath.exists():
                missing.append(str(filepath))
                continue
            self.tables[key] = pd.read_csv(filepath)
        if missing:
            raise FileNotFoundError("Не найдены файлы OULAD:\n" + "\n".join(missing))
        self._validate_schema()
        self._basic_cleanup()
        return self.tables

    def _validate_schema(self) -> None:
        rows = []
        for key, df in self.tables.items():
            required = REQUIRED_COLUMNS[key]
            missing_cols = [c for c in required if c not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"В таблице {key} отсутствуют обязательные колонки: {missing_cols}"
                )
            rows.append({
                "table": key,
                "rows": len(df),
                "columns": len(df.columns),
                "required_columns_ok": True,
            })
        pd.DataFrame(rows).to_csv(
            self.tables_dir / "schema_summary.csv", index=False, encoding="utf-8-sig"
        )

    def _basic_cleanup(self) -> None:
        self.tables["courses"]["module_presentation_length"] = pd.to_numeric(self.tables["courses"]["module_presentation_length"], errors="coerce")

        ass = self.tables["assessments"].copy()
        ass["date"] = pd.to_numeric(ass["date"], errors="coerce")
        ass["weight"] = pd.to_numeric(ass["weight"], errors="coerce")
        self.tables["assessments"] = ass

        reg = self.tables["studentRegistration"].copy()
        for col in ["date_registration", "date_unregistration"]:
            reg[col] = pd.to_numeric(reg[col], errors="coerce")
        self.tables["studentRegistration"] = reg

        sa = self.tables["studentAssessment"].copy()
        for col in ["date_submitted", "score", "is_banked"]:
            sa[col] = pd.to_numeric(sa[col], errors="coerce")
        self.tables["studentAssessment"] = sa

        sv = self.tables["studentVle"].copy()
        for col in ["date", "sum_click"]:
            sv[col] = pd.to_numeric(sv[col], errors="coerce")
        self.tables["studentVle"] = sv

        vle = self.tables["vle"].copy()
        for col in ["week_from", "week_to"]:
            vle[col] = pd.to_numeric(vle[col], errors="coerce")
        self.tables["vle"] = vle

    def build_student_base(self) -> pd.DataFrame:
        """Строит базовую таблицу студентов с демографией и сроками активности."""
        info = self.tables["studentInfo"].copy()
        reg = self.tables["studentRegistration"].copy()
        courses = self.tables["courses"].copy()

        base = info.merge(
            reg, on=["code_module", "code_presentation", "id_student"], how="left", validate="1:1",
        ).merge(courses, on=["code_module", "code_presentation"], how="left", validate="m:1")

        base = combine_keys(base, ["id_student", "code_module", "code_presentation"], "student_presentation_id")
        base["active_until_day"] = np.where(
            base["date_unregistration"].notna(),
            np.minimum(base["date_unregistration"], base["module_presentation_length"]),
            base["module_presentation_length"],
        )
        base["active_until_day"] = base["active_until_day"].fillna(base["module_presentation_length"])

        base["final_result_norm"] = (
            base["final_result"].astype(str).str.strip().str.lower().map(FINAL_RESULT_MAP).fillna(0.0)
        )
        self.student_base = base
        base.to_csv(self.tables_dir / "student_base.csv", index=False, encoding="utf-8-sig")
        return base

    def build_assessment_features(self) -> pd.DataFrame:
        """Вычисляет метрики выполнения тестов: сдачи, задержки, оценки."""
        if self.student_base is None:
            raise RuntimeError("Требуется предварительно выполнить build_student_base().")
        sa = self.tables["studentAssessment"].copy()
        base = self.student_base.copy()
        assessments = self.tables["assessments"].copy()

        assessments = assessments.merge(
            self.tables["courses"][["code_module", "code_presentation", "module_presentation_length"]],
            on=["code_module", "code_presentation"], how="left",
        )
        exam_mask = assessments["assessment_type"].astype(str).str.lower().eq("exam")
        assessments.loc[exam_mask & assessments["date"].isna(), "date"] = assessments.loc[
            exam_mask & assessments["date"].isna(), "module_presentation_length"
        ]

        expected = base[[
            "student_presentation_id", "id_student", "code_module", "code_presentation",
            "active_until_day", "final_result", "final_result_norm",
        ]].merge(
            assessments[[
                "code_module", "code_presentation", "id_assessment", "assessment_type", "date", "weight",
            ]],
            on=["code_module", "code_presentation"], how="left", validate="m:m",
        )
        expected = expected[expected["date"].fillna(np.inf) <= expected["active_until_day"].fillna(np.inf)].copy()
        expected = expected.merge(sa, on=["id_student", "id_assessment"], how="left", validate="1:1")

        expected["submitted"] = expected["score"].notna().astype(int)
        expected["delay_days"] = expected["date_submitted"] - expected["date"]
        expected["delay_days_clipped"] = expected["delay_days"].clip(lower=0)
        expected["on_time"] = np.where(
            expected["submitted"].eq(1) & expected["delay_days"].le(0), 1, 0,
        )
        expected["score_norm"] = expected["score"] / 100.0
        expected["weighted_score"] = expected["score_norm"].fillna(0.0) * expected["weight"].fillna(0.0)
        expected["weighted_score_possible"] = expected["weight"].fillna(0.0)
        expected["assessment_week"] = np.floor(expected["date"].fillna(0) / 7).astype(int)

        grouped = expected.groupby("student_presentation_id", dropna=False)
        feat = grouped.agg(
            expected_assessments=("id_assessment", "count"),
            submitted_assessments=("submitted", "sum"),
            ontime_assessments=("on_time", "sum"),
            mean_delay_days=("delay_days_clipped", "mean"),
            median_delay_days=("delay_days_clipped", "median"),
            max_delay_days=("delay_days_clipped", "max"),
            weighted_score_sum=("weighted_score", "sum"),
            weighted_score_possible=("weighted_score_possible", "sum"),
            banked_share=("is_banked", "mean"),
        ).reset_index()

        feat["missed_assessments"] = feat["expected_assessments"] - feat["submitted_assessments"]
        feat["submission_ratio"] = np.where(
            feat["expected_assessments"] > 0,
            feat["submitted_assessments"] / feat["expected_assessments"], 0.0,
        )
        feat["ontime_ratio"] = np.where(
            feat["submitted_assessments"] > 0,
            feat["ontime_assessments"] / feat["submitted_assessments"], 0.0,
        )
        feat["missed_ratio"] = np.where(
            feat["expected_assessments"] > 0,
            feat["missed_assessments"] / feat["expected_assessments"], 1.0,
        )
        feat["mastery_raw"] = np.where(
            feat["weighted_score_possible"] > 0,
            feat["weighted_score_sum"] / feat["weighted_score_possible"], 0.0,
        )
        feat["mastery_raw"] = feat["mastery_raw"].clip(0.0, 1.0)

        self.assessment_expected = expected
        self.assessment_features = feat
        expected.to_csv(self.tables_dir / "assessment_expected_events.csv", index=False, encoding="utf-8-sig")
        feat.to_csv(self.tables_dir / "assessment_features.csv", index=False, encoding="utf-8-sig")
        return feat

    def build_vle_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Вычисляет дневные и еженедельные метрики взаимодействия с материалами."""
        if self.student_base is None:
            raise RuntimeError("Требуется предварительно выполнить build_student_base().")
        sv = self.tables["studentVle"].copy()
        vle = self.tables["vle"].copy()
        base = self.student_base.copy()

        daily = sv.merge(
            vle[["id_site", "code_module", "code_presentation", "activity_type", "week_from", "week_to"]],
            on=["id_site", "code_module", "code_presentation"], how="left", validate="m:1",
        ).merge(
            base[[
                "student_presentation_id", "id_student", "code_module", "code_presentation",
                "active_until_day", "final_result", "module_presentation_length",
            ]],
            on=["id_student", "code_module", "code_presentation"], how="inner", validate="m:1",
        )
        daily = daily[daily["date"].fillna(np.inf) <= daily["active_until_day"].fillna(np.inf)].copy()
        daily["week_index"] = np.floor(daily["date"].fillna(0) / 7).astype(int)
        self.vle_daily = daily
        daily.to_csv(self.tables_dir / "vle_daily_events.csv", index=False, encoding="utf-8-sig")

        weekly = (
            daily.groupby(["student_presentation_id", "week_index"], dropna=False).agg(
                weekly_clicks=("sum_click", "sum"),
                weekly_active_days=("date", "nunique"),
                weekly_unique_sites=("id_site", "nunique"),
                weekly_unique_activity_types=("activity_type", "nunique"),
            ).reset_index()
        )
        self.vle_weekly = weekly
        weekly.to_csv(self.tables_dir / "vle_weekly_features.csv", index=False, encoding="utf-8-sig")
        return daily, weekly

    def build_student_level_features(self) -> pd.DataFrame:
        """Агрегирует оценки и взаимодействия на уровне студента, вычисляет четыре прокси-критерия."""
        if (
            self.student_base is None
            or self.assessment_features is None
            or self.vle_daily is None
            or self.vle_weekly is None
        ):
            raise RuntimeError("Требуется предварительно выполнить build_student_base, build_assessment_features и build_vle_features.")

        base = self.student_base.copy()
        daily = self.vle_daily.copy()
        weekly = self.vle_weekly.copy()
        assess = self.assessment_features.copy()

        vle_student = daily.groupby("student_presentation_id", dropna=False).agg(
            total_clicks=("sum_click", "sum"),
            active_days=("date", "nunique"),
            unique_sites=("id_site", "nunique"),
            unique_activity_types=("activity_type", "nunique"),
        ).reset_index()

        weekly_stats = weekly.groupby("student_presentation_id", dropna=False).agg(
            active_weeks=("week_index", "nunique"),
            mean_weekly_clicks=("weekly_clicks", "mean"),
            std_weekly_clicks=("weekly_clicks", "std"),
            mean_weekly_active_days=("weekly_active_days", "mean"),
        ).reset_index()

        base["available_weeks"] = np.ceil(
            base["active_until_day"].fillna(base["module_presentation_length"]).clip(lower=0) / 7.0
        ).astype(int)
        base["available_weeks"] = base["available_weeks"].replace(0, 1)

        features = (
            base.merge(vle_student, on="student_presentation_id", how="left")
                .merge(weekly_stats, on="student_presentation_id", how="left")
                .merge(assess, on="student_presentation_id", how="left")
        )

        fill_zero_cols = [
            "total_clicks", "active_days", "unique_sites", "unique_activity_types",
            "active_weeks", "mean_weekly_clicks", "std_weekly_clicks", "mean_weekly_active_days",
            "expected_assessments", "submitted_assessments", "ontime_assessments",
            "mean_delay_days", "median_delay_days", "max_delay_days",
            "weighted_score_sum", "weighted_score_possible", "banked_share",
            "missed_assessments", "submission_ratio", "ontime_ratio", "missed_ratio",
            "mastery_raw",
        ]
        for col in fill_zero_cols:
            if col in features.columns:
                features[col] = features[col].fillna(0.0)

        features["continuity_ratio"] = np.where(
            features["available_weeks"] > 0,
            features["active_weeks"] / features["available_weeks"], 0.0,
        ).clip(0.0, 1.0)

        features["eng_clicks_norm"] = minmax_series(features["total_clicks"])
        features["eng_active_days_norm"] = minmax_series(features["active_days"])
        features["eng_site_breadth_norm"] = minmax_series(features["unique_sites"])
        features["eng_activity_breadth_norm"] = minmax_series(features["unique_activity_types"])
        features["eng_regularity_norm"] = features["continuity_ratio"].clip(0.0, 1.0)

        features["Engagement"] = (
            self.weights.engagement_clicks * features["eng_clicks_norm"]
            + self.weights.engagement_active_days * features["eng_active_days_norm"]
            + self.weights.engagement_resource_breadth * features["eng_site_breadth_norm"]
            + self.weights.engagement_activity_breadth * features["eng_activity_breadth_norm"]
            + self.weights.engagement_regularity * features["eng_regularity_norm"]
        ).clip(0.0, 1.0)

        features["Mastery"] = features["mastery_raw"].clip(0.0, 1.0)

        features["sr_ontime_norm"] = features["ontime_ratio"].clip(0.0, 1.0)
        features["sr_delay_norm"] = robust_inverse_minmax(features["mean_delay_days"])
        features["sr_submission_norm"] = features["submission_ratio"].clip(0.0, 1.0)
        features["sr_continuity_norm"] = features["continuity_ratio"].clip(0.0, 1.0)

        features["SelfRegulation"] = (
            self.weights.selfreg_ontime * features["sr_ontime_norm"]
            + self.weights.selfreg_delay * features["sr_delay_norm"]
            + self.weights.selfreg_submission * features["sr_submission_norm"]
            + self.weights.selfreg_continuity * features["sr_continuity_norm"]
        ).clip(0.0, 1.0)

        features["Outcome"] = (
            self.weights.outcome_mastery * features["Mastery"]
            + self.weights.outcome_final_result * features["final_result_norm"]
        ).clip(0.0, 1.0)

        features["completed_module"] = ~features["final_result"].astype(str).str.lower().isin(["withdrawn", "withdrawal"])
        features["passed_module"] = features["final_result"].astype(str).str.lower().isin(["pass", "distinction"])

        self.student_features = features
        features.to_csv(
            self.dataset_dir / "oulad_student_presentation_features.csv",
            index=False, encoding="utf-8-sig",
        )
        return features

    def build_weekly_trajectory_dataset(self) -> pd.DataFrame:
        """Формирует траектории еженедельных метрик для каждого студента."""
        if self.student_features is None or self.vle_weekly is None:
            raise RuntimeError("Требуется предварительно выполнить build_student_level_features().")
        base = self.student_features.copy()
        weekly = self.vle_weekly.copy()

        weekly = weekly.sort_values(["student_presentation_id", "week_index"]).copy()
        weekly["cum_clicks"] = weekly.groupby("student_presentation_id")["weekly_clicks"].cumsum()
        weekly["cum_active_days"] = weekly.groupby("student_presentation_id")["weekly_active_days"].cumsum()
        weekly["cum_unique_sites_proxy"] = weekly.groupby("student_presentation_id")["weekly_unique_sites"].cumsum()
        weekly["cum_unique_activity_types_proxy"] = weekly.groupby("student_presentation_id")["weekly_unique_activity_types"].cumsum()

        available_weeks = base[["student_presentation_id", "available_weeks"]]
        weekly = weekly.merge(available_weeks, on="student_presentation_id", how="left")
        weekly["trajectory_progress"] = np.where(
            weekly["available_weeks"] > 0,
            (weekly["week_index"] + 1) / weekly["available_weeks"], 0.0,
        ).clip(0.0, 1.0)

        weekly = weekly.merge(
            base[[
                "student_presentation_id", "code_module", "code_presentation", "id_student",
                "final_result", "Outcome", "Mastery", "Engagement", "SelfRegulation",
                "final_result_norm", "num_of_prev_attempts", "studied_credits",
                "gender", "highest_education", "imd_band", "age_band", "disability",
            ]],
            on="student_presentation_id", how="left",
        )
        weekly.to_csv(
            self.dataset_dir / "oulad_weekly_trajectory_features.csv",
            index=False, encoding="utf-8-sig",
        )
        return weekly

    def run_all(self) -> Dict[str, Any]:
        self.load_tables()
        self.build_student_base()
        self.build_assessment_features()
        self.build_vle_features()
        self.build_student_level_features()
        self.build_weekly_trajectory_dataset()
        summary = {
            "n_student_presentations": int(len(self.student_features)) if self.student_features is not None else 0,
            "n_students": int(self.student_features["id_student"].nunique()) if self.student_features is not None else 0,
            "n_weeks_rows": int(len(self.vle_weekly)) if self.vle_weekly is not None else 0,
        }
        with open(self.tables_dir / "summary_stats.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary


# ---------------------------------------------------------------------------
# Mixed-step catalog
# ---------------------------------------------------------------------------


def _bucket_delay(delay: float) -> str:
    if delay <= 0:
        return "on_time"
    if delay <= 3:
        return "slight"
    if delay <= 7:
        return "late"
    return "very_late"


def build_mixed_step_catalog(
    vle_daily: pd.DataFrame,
    assessments_expected: pd.DataFrame,
    *,
    mode: str = "mixed",
    top_activity_types: int = 6,
    top_assessment_types: int = 3,
    bucket_delay: bool = True,
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]]]:
    """Строит каталог действий (item) для OULAD на основе активности взаимодействия и оценок.

    Возвращает DataFrame с полями item_id, kind, key и словарь items_meta для быстрого доступа.
    """
    catalog_rows: List[Dict[str, Any]] = []

    if mode in ("mixed", "resource_only") and vle_daily is not None and not vle_daily.empty:
        activity_counts = vle_daily["activity_type"].value_counts()
        top_types = list(activity_counts.head(top_activity_types).index)
        for atype in top_types:
            catalog_rows.append({
                "kind": "vle",
                "key": f"vle__{atype}",
                "activity_type": atype,
                "assessment_type": None,
                "delay_bucket": None,
            })

    if mode in ("mixed", "assessment_only") and assessments_expected is not None and not assessments_expected.empty:
        ass_counts = assessments_expected["assessment_type"].value_counts()
        top_ass = list(ass_counts.head(top_assessment_types).index)
        for atype in top_ass:
            if bucket_delay:
                for bucket in ["on_time", "slight", "late", "very_late"]:
                    catalog_rows.append({
                        "kind": "assessment",
                        "key": f"assessment__{atype}__{bucket}",
                        "activity_type": None,
                        "assessment_type": atype,
                        "delay_bucket": bucket,
                    })
            else:
                catalog_rows.append({
                    "kind": "assessment",
                    "key": f"assessment__{atype}",
                    "activity_type": None,
                    "assessment_type": atype,
                    "delay_bucket": None,
                })

    if not catalog_rows:
        raise ValueError("Не удалось собрать каталог OULAD — проверьте исходные данные и режим.")

    catalog = pd.DataFrame(catalog_rows).reset_index(drop=True)
    catalog.insert(0, "item_id", catalog.index.astype(int))
    meta = {int(row.item_id): row.to_dict() for _, row in catalog.iterrows()}
    return catalog, meta


# ---------------------------------------------------------------------------
# Step-level dataset
# ---------------------------------------------------------------------------


def build_step_level_dataframe(
    weekly_df: pd.DataFrame,
    vle_daily: pd.DataFrame,
    assessments_expected: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    bucket_delay: bool = True,
) -> pd.DataFrame:
    """Собирает таблицу на уровне шага (user-item-proxy) для обучения DeepFM.

    Каждый шаг соответствует одной неделе и одному типу контента.
    Значения критериев вычисляются как еженедельные агрегаты, нормализованные в диапазон [0, 1].
    """
    rows: List[Dict[str, Any]] = []

    vle_by_week = (
        vle_daily.groupby(["student_presentation_id", "week_index", "activity_type"], dropna=False)
        .agg(
            clicks=("sum_click", "sum"),
            active_days=("date", "nunique"),
            unique_sites=("id_site", "nunique"),
        )
        .reset_index()
    )

    ass_events = assessments_expected.copy()
    ass_events["delay_bucket"] = ass_events["delay_days_clipped"].fillna(8).apply(_bucket_delay) if bucket_delay else "any"
    ass_by_week = (
        ass_events.groupby([
            "student_presentation_id", "assessment_week", "assessment_type", "delay_bucket",
        ], dropna=False)
        .agg(
            submitted=("submitted", "sum"),
            on_time=("on_time", "sum"),
            mean_weighted=("weighted_score", "mean"),
            n_events=("id_assessment", "count"),
        )
        .reset_index()
    )

    catalog_by_key = {(row.kind, row.key): int(row.item_id) for _, row in catalog.iterrows()}
    max_clicks = max(1.0, float(vle_by_week["clicks"].max() if not vle_by_week.empty else 1.0))
    max_days = max(1.0, float(vle_by_week["active_days"].max() if not vle_by_week.empty else 1.0))

    weekly_context = weekly_df.set_index(["student_presentation_id", "week_index"])

    for _, row in vle_by_week.iterrows():
        key = ("vle", f"vle__{row.activity_type}")
        if key not in catalog_by_key:
            continue
        item_id = catalog_by_key[key]
        sp_id = row.student_presentation_id
        week = int(row.week_index)
        ctx = weekly_context.loc[(sp_id, week)] if (sp_id, week) in weekly_context.index else None
        clicks_norm = min(1.0, float(row.clicks) / max_clicks)
        days_norm = min(1.0, float(row.active_days) / max_days)
        breadth_norm = min(1.0, float(row.unique_sites) / 5.0)
        engagement = float(np.clip(0.6 * clicks_norm + 0.3 * days_norm + 0.1 * breadth_norm, 0.0, 1.0))
        rows.append({
            "student_presentation_id": sp_id,
            "id_student": ctx["id_student"] if ctx is not None else None,
            "week_index": week,
            "item_id": item_id,
            "kind": "vle",
            "mastery": float(ctx["Mastery"]) if ctx is not None else 0.0,
            "engagement": engagement,
            "selfregulation": float(ctx["SelfRegulation"]) if ctx is not None else 0.0,
            "outcome": float(ctx["Outcome"]) if ctx is not None else 0.0,
        })

    for _, row in ass_by_week.iterrows():
        bucket = row.delay_bucket if bucket_delay else None
        key_specific = ("assessment", f"assessment__{row.assessment_type}__{bucket}") if bucket_delay else ("assessment", f"assessment__{row.assessment_type}")
        if key_specific not in catalog_by_key:
            continue
        item_id = catalog_by_key[key_specific]
        sp_id = row.student_presentation_id
        week = int(row.assessment_week)
        ctx = weekly_context.loc[(sp_id, week)] if (sp_id, week) in weekly_context.index else None
        submitted_ratio = float(row.submitted / max(row.n_events, 1))
        ontime_ratio = float(row.on_time / max(row.n_events, 1))
        mean_weighted = float(row.mean_weighted or 0.0)
        mastery = float(np.clip(mean_weighted, 0.0, 1.0))
        selfregulation = float(np.clip(0.6 * ontime_ratio + 0.4 * submitted_ratio, 0.0, 1.0))
        outcome = float(np.clip(0.7 * mastery + 0.3 * ontime_ratio, 0.0, 1.0))
        engagement = float(np.clip(ctx["Engagement"] if ctx is not None else 0.0, 0.0, 1.0))
        rows.append({
            "student_presentation_id": sp_id,
            "id_student": ctx["id_student"] if ctx is not None else None,
            "week_index": week,
            "item_id": item_id,
            "kind": "assessment",
            "mastery": mastery,
            "engagement": engagement,
            "selfregulation": selfregulation,
            "outcome": outcome,
        })

    step_df = pd.DataFrame(rows)
    return step_df


# ---------------------------------------------------------------------------
# Бандл
# ---------------------------------------------------------------------------


def _derive_context_encoder(values: pd.Series) -> Tuple[pd.Series, Dict[str, int]]:
    categories = {val: i for i, val in enumerate(sorted(values.dropna().astype(str).unique()))}
    encoded = values.astype(str).map(categories).fillna(0).astype(int)
    return encoded, categories


def build_oulad_bundle(config: Optional[Mapping[str, Any]] = None) -> DatasetBundle:
    """Строит DatasetBundle для OULAD.

    Этапы:
    1. Выполняет OULADAnalyzerPreprocessor.run_all(), если обработанные данные отсутствуют
       или force_rebuild=True в конфигурации.
    2. Строит каталог действий на основе выбранного режима.
    3. Собирает данные на уровне шага и формирует DatasetBundle.
    """
    cfg = dict(config or {})
    dataset_cfg = dict(cfg.get("dataset", {}))
    raw_dir = Path(dataset_cfg.get("raw_dir", "data/raw/oulad"))
    processed_dir = Path(dataset_cfg.get("processed_dir", "data/processed/oulad"))
    force_rebuild = bool(dataset_cfg.get("force_rebuild", False))
    catalog_cfg = dict(dataset_cfg.get("catalog", {}))
    catalog_mode = catalog_cfg.get("mode", "mixed")
    top_activity = int(catalog_cfg.get("top_activity_types", 6))
    top_assessment = int(catalog_cfg.get("top_assessment_types", 3))
    bucket_delay = bool(catalog_cfg.get("bucket_delay", True))

    weights = ProxyWeights.from_mapping(dataset_cfg.get("proxy_weights", {}))

    expected_student_csv = processed_dir / "prepared_datasets" / "oulad_student_presentation_features.csv"
    expected_weekly_csv = processed_dir / "prepared_datasets" / "oulad_weekly_trajectory_features.csv"
    expected_assessment_csv = processed_dir / "tables" / "assessment_expected_events.csv"
    expected_vle_daily_csv = processed_dir / "tables" / "vle_daily_events.csv"

    needs_build = force_rebuild or not all(
        p.exists() for p in (
            expected_student_csv, expected_weekly_csv,
            expected_assessment_csv, expected_vle_daily_csv,
        )
    )

    if needs_build:
        logger.info("Сборка OULAD-пайплайна: raw=%s, processed=%s", raw_dir, processed_dir)
        prep = OULADAnalyzerPreprocessor(raw_dir, processed_dir, weights=weights)
        prep.run_all()

    student_features = pd.read_csv(expected_student_csv)
    weekly = pd.read_csv(expected_weekly_csv)
    assessments_expected = pd.read_csv(expected_assessment_csv)
    vle_daily = pd.read_csv(expected_vle_daily_csv)

    catalog, items_meta = build_mixed_step_catalog(
        vle_daily=vle_daily,
        assessments_expected=assessments_expected,
        mode=catalog_mode,
        top_activity_types=top_activity,
        top_assessment_types=top_assessment,
        bucket_delay=bucket_delay,
    )

    step_df = build_step_level_dataframe(
        weekly_df=weekly,
        vle_daily=vle_daily,
        assessments_expected=assessments_expected,
        catalog=catalog,
        bucket_delay=bucket_delay,
    )

    if step_df.empty:
        raise RuntimeError("step_df пустой: проверьте корректность исходных OULAD-файлов.")

    user_ids = sorted(step_df["student_presentation_id"].dropna().unique())
    user_encoder = {u: idx for idx, u in enumerate(user_ids)}
    # Удаляем строки со студентами, которых нет в энкодере (на случай NaN)
    step_df = step_df[step_df["student_presentation_id"].isin(user_encoder.keys())].copy()
    step_df["UserID_encoded"] = step_df["student_presentation_id"].map(user_encoder).astype(int)
    step_df.rename(columns={"item_id": "ItemID_encoded"}, inplace=True)

    ratings = step_df[[
        "UserID_encoded", "ItemID_encoded", "week_index",
        "mastery", "engagement", "selfregulation", "outcome",
    ]].rename(columns={
        "mastery": "Mastery",
        "engagement": "Engagement",
        "selfregulation": "SelfRegulation",
        "outcome": "Outcome",
    })

    # Контекст: модуль курса и сессию представления.
    students_ctx = (
        student_features[["student_presentation_id", "code_module", "code_presentation", "age_band", "gender", "disability"]]
        .dropna(subset=["student_presentation_id"])
    )
    users_df = students_ctx.drop_duplicates("student_presentation_id").copy()
    # Фильтруем студентов, которые есть в кодировщике пользователей
    users_df = users_df[users_df["student_presentation_id"].isin(user_encoder.keys())].copy()
    users_df["UserID_encoded"] = users_df["student_presentation_id"].map(user_encoder).astype(int)

    module_enc, module_map = _derive_context_encoder(users_df["code_module"])
    pres_enc, pres_map = _derive_context_encoder(users_df["code_presentation"])
    users_df["Module_encoded"] = module_enc
    users_df["Presentation_encoded"] = pres_enc

    # Добавляем контекст в каждую строку таблицы оценок
    ratings = ratings.merge(
        users_df[["UserID_encoded", "Module_encoded", "Presentation_encoded"]],
        on="UserID_encoded", how="left",
    )

    items_df = catalog.copy()

    # Вычисляем популярность элементов каталога для метрики новизны
    popularity = ratings["ItemID_encoded"].value_counts().to_dict()

    n_users = int(len(user_encoder))
    n_items = int(len(catalog))

    # Размер состояния DQN: пользовательское вложение, контекст курса/сессии,
    # прогресс, демография, история взаимодействий, временные метки и вспомогательные признаки
    state_dim = int(cfg.get("model", {}).get("dqn", {}).get("state_dim", 96))

    bundle = DatasetBundle(
        dataset_type="oulad",
        ratings=ratings,
        users=users_df,
        items=items_df,
        target_columns=["Mastery", "Engagement", "SelfRegulation", "Outcome"],
        n_users=n_users,
        n_items=n_items,
        state_dim=state_dim,
        context_columns=["Module_encoded", "Presentation_encoded"],
        context_sizes=[max(1, len(module_map)), max(1, len(pres_map))],
        encoders={"user": user_encoder, "module": module_map, "presentation": pres_map},
        scalers={},
        item_popularity=popularity,
        trajectories=None,
        action_masks=None,
        metadata={
            "raw_dir": str(raw_dir),
            "processed_dir": str(processed_dir),
            "catalog_mode": catalog_mode,
            "catalog": catalog,
            "items_meta": items_meta,
            "student_features": student_features,
            "weekly": weekly,
            "assessments_expected": assessments_expected,
            "vle_daily_summary": vle_daily.groupby("activity_type")["sum_click"].sum().to_dict(),
            "final_result_map": FINAL_RESULT_MAP,
        },
    )
    logger.info("OULAD bundle: users=%d, items=%d, step_rows=%d", n_users, n_items, len(ratings))
    return bundle
