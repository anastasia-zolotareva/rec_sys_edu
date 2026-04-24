#!/usr/bin/env python
"""
Скрипт-обёртка для обучения Dueling DQN через новый CLI.

Реальная логика находится в :mod:`src.cli`. Этот файл оставлен для обратной
совместимости с историческим способом запуска.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.cli import main as cli_main  # noqa: E402


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение DQN агента (обёртка над CLI)")
    parser.add_argument("--dataset", choices=["itmrec", "oulad"], default="itmrec")
    parser.add_argument("--config", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--deepfm-checkpoint", default=None)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse()
    cli_args = ["train", "dqn", "--dataset", args.dataset]
    if args.config:
        cli_args += ["--config", args.config]
    if args.episodes is not None:
        cli_args += ["--episodes", str(args.episodes)]
    if args.deepfm_checkpoint:
        cli_args += ["--deepfm-checkpoint", args.deepfm_checkpoint]
    if args.run_name:
        cli_args += ["--run-name", args.run_name]
    return cli_main(cli_args)


if __name__ == "__main__":
    sys.exit(main())
