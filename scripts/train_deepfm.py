#!/usr/bin/env python
"""Тонкая обёртка для обучения DeepFM+SVD++ через CLI.

Примеры:
    python scripts/train_deepfm.py --dataset itmrec --config configs/itmrec.yaml
    python scripts/train_deepfm.py --dataset oulad  --config configs/oulad.yaml --epochs 30
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.cli import main


if __name__ == "__main__":
    # Для обратной совместимости: по умолчанию ITM-Rec + дефолтный YAML.
    args = sys.argv[1:]
    if "--dataset" not in args:
        args = ["--dataset", "itmrec", "--config", "configs/itmrec.yaml", *args]
    sys.exit(main(["train", "static", *args]))
