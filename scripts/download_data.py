#!/usr/bin/env python
"""Тонкая обёртка — загрузка ITM-Rec или OULAD через CLI.

Примеры:
    python scripts/download_data.py --dataset itmrec
    python scripts/download_data.py --dataset oulad
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.cli import main


if __name__ == "__main__":
    dataset = "itmrec"
    extra = []
    if len(sys.argv) > 1 and sys.argv[1].startswith("--dataset"):
        pass  # всё передано пользователем
    else:
        # Обратная совместимость: скрипт без аргументов по-старому грузит ITM-Rec
        extra = ["--dataset", "itmrec"]
    sys.exit(main(["data", "download", *extra, *sys.argv[1:]]))
