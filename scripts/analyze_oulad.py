#!/usr/bin/env python
"""Тонкая обёртка для OULAD EDA через CLI.

Запускает ``python -m src.cli data analyze --dataset oulad`` и
формирует CSV-таблицы и PNG-графики в ``data/processed/oulad``
(или в пользовательском каталоге ``--output-dir``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.cli import main as cli_main  # noqa: E402


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OULAD EDA (обёртка над CLI)")
    parser.add_argument("--config", default="configs/oulad.yaml")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse()
    cli_args = ["data", "analyze", "--dataset", "oulad", "--config", args.config]
    if args.output_dir:
        cli_args += ["--output-dir", args.output_dir]
    return cli_main(cli_args)


if __name__ == "__main__":
    sys.exit(main())
