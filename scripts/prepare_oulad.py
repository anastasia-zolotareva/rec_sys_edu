#!/usr/bin/env python
"""
Скрипт-обёртка для подготовки OULAD через новый CLI.

Использует ``src.cli data prepare --dataset oulad`` плюс аргумент ``--config``
(по умолчанию ``configs/oulad.yaml``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.cli import main as cli_main  # noqa: E402


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Подготовка OULAD (обёртка над CLI)")
    parser.add_argument("--config", default="configs/oulad.yaml")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse()
    cli_args = ["data", "prepare", "--dataset", "oulad", "--config", args.config]
    if args.output_dir:
        cli_args += ["--output-dir", args.output_dir]
    return cli_main(cli_args)


if __name__ == "__main__":
    sys.exit(main())
