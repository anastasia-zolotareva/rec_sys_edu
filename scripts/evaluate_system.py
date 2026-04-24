#!/usr/bin/env python
"""Тонкая обёртка для запуска оценки системы через CLI.

Примеры::

    python scripts/evaluate_system.py --dataset itmrec \
        --deepfm-checkpoint data/models/deepfm_itmrec_best.pth \
        --dqn-checkpoint data/models/dqn_itmrec_best.pth

    python scripts/evaluate_system.py --dataset oulad --hypothesis H1 \
        --deepfm-checkpoint data/models/deepfm_oulad_best.pth \
        --dqn-checkpoint data/models/dqn_oulad_best.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.cli import main as cli_main  # noqa: E402


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Оценка системы (обёртка над CLI)")
    parser.add_argument("--dataset", choices=["itmrec", "oulad"], default="itmrec")
    parser.add_argument("--config", default=None)
    parser.add_argument("--hypothesis", choices=["H1", "H2", "H3", "all"], default="all")
    parser.add_argument("--deepfm-checkpoint", default=None)
    parser.add_argument("--dqn-checkpoint", default=None)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse()
    cli_args = [
        "evaluate",
        "--dataset", args.dataset,
        "--hypothesis", args.hypothesis,
    ]
    if args.config:
        cli_args += ["--config", args.config]
    if args.deepfm_checkpoint:
        cli_args += ["--deepfm-checkpoint", args.deepfm_checkpoint]
    if args.dqn_checkpoint:
        cli_args += ["--dqn-checkpoint", args.dqn_checkpoint]
    if args.run_name:
        cli_args += ["--run-name", args.run_name]
    return cli_main(cli_args)


if __name__ == "__main__":
    sys.exit(main())
