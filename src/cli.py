"""
Единая точка входа через командную строку.

Типичные сценарии:

.. code-block:: bash

    python -m src.cli data download --dataset itmrec
    python -m src.cli data download --dataset oulad --data-dir data/raw/oulad
    python -m src.cli data prepare --dataset oulad --output-dir data/processed/oulad

    python -m src.cli train static --dataset itmrec --config configs/itmrec.yaml
    python -m src.cli train dqn    --dataset itmrec --config configs/itmrec.yaml

    python -m src.cli evaluate     --dataset itmrec --config configs/itmrec.yaml

На этапе 1 реализованы подкоманды для ITM-Rec; OULAD и DQN активируются
по мере реализации соответствующих компонентов.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import api

logger = logging.getLogger("rec_sys_edu")


# ---------------------------------------------------------------------------
# Парсер
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rec-sys-edu",
        description="Гибридная рекомендательная система DeepFM+SVD++ / Dueling DQN",
    )
    sub = parser.add_subparsers(dest="group", required=True)

    # --- data ---
    p_data = sub.add_parser("data", help="Работа с датасетами")
    sub_data = p_data.add_subparsers(dest="command", required=True)

    p_data_dl = sub_data.add_parser("download", help="Загрузить датасет")
    p_data_dl.add_argument("--dataset", choices=["itmrec", "oulad"], required=True)
    p_data_dl.add_argument("--output-dir", default=None)
    p_data_dl.add_argument("--kaggle-name", default=None)

    p_data_prep = sub_data.add_parser("prepare", help="Построить DatasetBundle и сохранить сводки")
    p_data_prep.add_argument("--dataset", choices=["itmrec", "oulad"], required=True)
    p_data_prep.add_argument("--config", default=None, help="Путь к YAML-конфигу")
    p_data_prep.add_argument("--output-dir", default=None, help="Куда писать сводки")

    p_data_analyze = sub_data.add_parser(
        "analyze",
        help="Провести EDA (таблицы и графики) для датасета (сейчас - OULAD)",
    )
    p_data_analyze.add_argument("--dataset", choices=["oulad"], required=True)
    p_data_analyze.add_argument("--config", default=None, help="Путь к YAML-конфигу")
    p_data_analyze.add_argument("--output-dir", default=None, help="Куда писать отчеты и графики")

    # --- train ---
    p_train = sub.add_parser("train", help="Обучение моделей")
    sub_train = p_train.add_subparsers(dest="command", required=True)

    p_train_static = sub_train.add_parser("static", help="Обучить DeepFM+SVD++ (статическая модель)")
    p_train_static.add_argument("--dataset", choices=["itmrec", "oulad"], required=True)
    p_train_static.add_argument("--config", default=None)
    p_train_static.add_argument("--epochs", type=int, default=None)
    p_train_static.add_argument("--batch-size", type=int, default=None)
    p_train_static.add_argument("--lr", type=float, default=None)
    p_train_static.add_argument("--run-name", default=None)

    p_train_dqn = sub_train.add_parser("dqn", help="Обучить Dueling DQN поверх DeepFM")
    p_train_dqn.add_argument("--dataset", choices=["itmrec", "oulad"], required=True)
    p_train_dqn.add_argument("--config", default=None)
    p_train_dqn.add_argument("--episodes", type=int, default=None)
    p_train_dqn.add_argument("--deepfm-checkpoint", default=None)
    p_train_dqn.add_argument("--run-name", default=None)

    # --- evaluate ---
    p_eval = sub.add_parser("evaluate", help="Запустить оценку / гипотезы")
    p_eval.add_argument("--dataset", choices=["itmrec", "oulad"], required=True)
    p_eval.add_argument("--config", default=None)
    p_eval.add_argument("--hypothesis", choices=["H1", "H2", "H3", "all"], default="all")
    p_eval.add_argument("--deepfm-checkpoint", default=None)
    p_eval.add_argument("--dqn-checkpoint", default=None)
    p_eval.add_argument("--run-name", default=None)

    return parser


# ---------------------------------------------------------------------------
# Хендлеры
# ---------------------------------------------------------------------------


def _build_overrides_for_training(args: argparse.Namespace) -> dict:
    overrides: dict = {}
    if getattr(args, "epochs", None) is not None:
        overrides.setdefault("model", {}).setdefault("deepfm", {})["n_epochs"] = args.epochs
    if getattr(args, "batch_size", None) is not None:
        overrides.setdefault("model", {}).setdefault("deepfm", {})["batch_size"] = args.batch_size
    if getattr(args, "lr", None) is not None:
        overrides.setdefault("model", {}).setdefault("deepfm", {})["lr"] = args.lr
    if getattr(args, "episodes", None) is not None:
        overrides.setdefault("training", {}).setdefault("dqn", {})["n_episodes"] = args.episodes
    return overrides


def _cmd_data_download(args: argparse.Namespace) -> int:
    if args.dataset == "itmrec":
        from .data.loaders import download_kaggle_dataset
        kaggle = args.kaggle_name or "irecsys/itmrec"
        out = args.output_dir or "data/raw"
        download_kaggle_dataset(kaggle, out)
    else:
        from .data.loaders import download_oulad_dataset
        kaggle = args.kaggle_name or "anlgrbz/student-demographics-online-education-dataoulad"
        out = args.output_dir or "data/raw/oulad"
        download_oulad_dataset(kaggle, out)
    return 0


def _cmd_data_prepare(args: argparse.Namespace) -> int:
    config = api.build_config(args.dataset, yaml_path=args.config)
    if args.output_dir:
        config.setdefault("dataset", {})["processed_dir"] = args.output_dir
    bundle = api.load_dataset_bundle(args.dataset, config=config)
    print("Dataset bundle:")
    for k, v in bundle.describe().items():
        print(f"  {k}: {v}")
    return 0


def _cmd_data_analyze(args: argparse.Namespace) -> int:
    if args.dataset != "oulad":
        print("На текущий момент data analyze поддерживает только OULAD.", file=sys.stderr)
        return 2

    from .data.preprocess_oulad import OULADAnalyzerPreprocessor
    from .data.oulad_reports import run_full_analysis

    config = api.build_config(args.dataset, yaml_path=args.config)
    ds_cfg = config.get("dataset", {})
    raw_dir = ds_cfg.get("raw_dir", "data/raw/oulad")
    output_dir = args.output_dir or ds_cfg.get("analysis_dir") or "data/processed/oulad"

    analyzer = OULADAnalyzerPreprocessor(raw_dir=raw_dir, output_dir=output_dir)
    summary = run_full_analysis(analyzer)
    print("OULAD EDA завершён. Основные показатели:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"Таблицы: {analyzer.tables_dir}")
    print(f"Графики: {analyzer.fig_dir}")
    return 0


def _cmd_train_static(args: argparse.Namespace) -> int:
    overrides = _build_overrides_for_training(args)
    config = api.build_config(args.dataset, yaml_path=args.config, overrides=overrides)
    run_dir = api.prepare_run(config, run_name=args.run_name or f"{args.dataset}_deepfm")
    result = api.train_static(args.dataset, config=config, run_dir=run_dir)
    print(f"Лучшая валидационная ошибка: {result['history'].get('best_val_loss'):.4f}")
    print(f"Чекпоинт: {result['history'].get('best_checkpoint')}")
    return 0


def _cmd_train_dqn(args: argparse.Namespace) -> int:
    overrides = _build_overrides_for_training(args)
    config = api.build_config(args.dataset, yaml_path=args.config, overrides=overrides)
    run_dir = api.prepare_run(config, run_name=args.run_name or f"{args.dataset}_dqn")

    deepfm_checkpoint = args.deepfm_checkpoint
    if deepfm_checkpoint is None:
        # Попытка найти последний чекпоинт DeepFM в стандартных местах.
        prefix = config.get("artifacts", {}).get("run_prefix", args.dataset)
        candidates = [
            Path(config.get("artifacts", {}).get("models_dir", "data/models"))
            / f"deepfm_{prefix}_best.pth",
            Path("data/models") / "deepfm_svdplusplus_best.pth",
        ]
        for cand in candidates:
            if cand.exists():
                deepfm_checkpoint = str(cand)
                break
        if deepfm_checkpoint is None:
            print(
                "Не указан --deepfm-checkpoint и стандартные пути не найдены. "
                "Сначала запустите train static.",
                file=sys.stderr,
            )
            return 2

    result = api.train_dqn(
        args.dataset,
        config=config,
        run_dir=run_dir,
        deepfm_checkpoint=deepfm_checkpoint,
    )
    history = result["history"]
    final = history.get("final_evaluation", {})
    print(f"DQN-чекпоинт: {result['checkpoint']}")
    if final:
        print(f"Финальная оценка: mean_reward={final.get('mean_reward'):.3f} ± {final.get('std_reward'):.3f}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    config = api.build_config(args.dataset, yaml_path=args.config)
    run_dir = api.prepare_run(config, run_name=args.run_name or f"{args.dataset}_eval")

    deepfm_checkpoint = args.deepfm_checkpoint
    dqn_checkpoint = args.dqn_checkpoint
    if not deepfm_checkpoint or not dqn_checkpoint:
        print(
            "Для запуска оценки нужно указать --deepfm-checkpoint и --dqn-checkpoint.",
            file=sys.stderr,
        )
        return 2

    result = api.evaluate_system(
        args.dataset,
        hypothesis=args.hypothesis,
        config=config,
        run_dir=run_dir,
        deepfm_checkpoint=deepfm_checkpoint,
        dqn_checkpoint=dqn_checkpoint,
    )
    print(f"Результаты оценки: {run_dir / 'tables' / 'evaluation_summary.json'}")
    for key in ("H1", "H2", "H3"):
        if key in result["results"]:
            print(f"  {key}: сохранено")
    return 0


COMMANDS = {
    ("data", "download"): _cmd_data_download,
    ("data", "prepare"): _cmd_data_prepare,
    ("data", "analyze"): _cmd_data_analyze,
    ("train", "static"): _cmd_train_static,
    ("train", "dqn"): _cmd_train_dqn,
    ("evaluate", None): _cmd_evaluate,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command_key = (args.group, getattr(args, "command", None))
    handler = COMMANDS.get(command_key)
    if handler is None:
        parser.error(f"Неизвестная команда: {command_key}")
        return 2
    return handler(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
