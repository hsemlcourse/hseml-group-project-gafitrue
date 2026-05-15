"""Глобальная конфигурация проекта.

Здесь зафиксирован SEED и все пути. Импортируется во всех скриптах,
чтобы избежать рассинхрона между EDA, обучением и инференсом.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# --- Воспроизводимость ----------------------------------------------------
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Фиксирует все известные источники случайности.

    Вызывать в начале каждого entrypoint (eda.py, train.py, app.py).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# --- Пути -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw" / "cve.csv"
DATA_PROCESSED = ROOT / "data" / "processed" / "cve_processed.parquet"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
REPORT_IMAGES = ROOT / "report" / "images"


# --- Целевая переменная ---------------------------------------------------
SEVERITY_MAP = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"}

# CVSS v2 границы по официальной спецификации FIRST.
# https://www.first.org/cvss/v2/guide#3-2-Rating-Scale
CVSS_BINS = [-0.01, 4.0, 7.0, 9.0, 10.01]
CVSS_LABELS = [0, 1, 2, 3]
