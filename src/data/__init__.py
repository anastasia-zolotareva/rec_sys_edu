"""
Модуль работы с данными.

Включает:
- Загрузку данных (Kaggle / локально) для ITM-Rec и OULAD.
- Предобработку и построение единого DatasetBundle.
- PyTorch Dataset для ITM-REC.
"""

from __future__ import annotations

try:
    from .dataset import ITMDataset
    from .loaders import (
        download_kaggle_dataset,
        download_oulad_dataset,
        load_all_data,
        load_dataset,
        load_group_ratings,
        load_items,
        load_ratings,
        load_users,
        verify_oulad_files,
    )
    from .preprocess_itmrec import build_itmrec_bundle
    from .oulad_reports import (
        create_all_figures as oulad_create_all_figures,
        export_report_tables as oulad_export_report_tables,
        run_full_analysis as oulad_run_full_analysis,
        save_summary_tables as oulad_save_summary_tables,
    )
    from .preprocess_oulad import (
        OULADAnalyzerPreprocessor,
        ProxyWeights,
        build_mixed_step_catalog,
        build_oulad_bundle,
        build_step_level_dataframe,
    )
    from .preprocessing import (
        encode_categorical,
        fill_missing_values,
        normalize_ratings,
        validate_data,
    )
    from .schemas import DatasetBundle

    __all__ = [
        "ITMDataset",
        "DatasetBundle",
        "OULADAnalyzerPreprocessor",
        "ProxyWeights",
        "build_itmrec_bundle",
        "build_mixed_step_catalog",
        "build_oulad_bundle",
        "build_step_level_dataframe",
        "oulad_create_all_figures",
        "oulad_export_report_tables",
        "oulad_run_full_analysis",
        "oulad_save_summary_tables",
        "download_kaggle_dataset",
        "download_oulad_dataset",
        "verify_oulad_files",
        "load_ratings",
        "load_users",
        "load_items",
        "load_group_ratings",
        "load_all_data",
        "load_dataset",
        "fill_missing_values",
        "encode_categorical",
        "normalize_ratings",
        "validate_data",
    ]
except ImportError:
    __all__ = []
