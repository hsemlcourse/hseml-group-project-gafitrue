"""Загрузка, очистка и feature engineering для CVE-датасета.

Закрывает замечания CP1:
* выбросы по числовым признакам обрабатываются явно (IQR + клиппинг);
* таргет строится по официальным границам CVSS v2 (FIRST);
* CWE-таксономия используется как категориальный признак;
* train/val/test разбиваются по году публикации, чтобы исключить data leakage.

Защита от data leakage
----------------------
Колонки access_authentication, access_complexity, access_vector,
impact_availability, impact_confidentiality, impact_integrity — это
компоненты CVSS v2 metric vector. По формуле CVSS v2 они однозначно
определяют CVSS score. Поскольку таргет severity получен из CVSS,
использование этих колонок — прямая утечка. Они НЕ попадают в feature
pipeline (см. src/features.py); удерживаются в DataFrame только для EDA.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import CVSS_BINS, CVSS_LABELS, DATA_RAW, SEED  # noqa: F401

KEYWORDS = [
    "remote",
    "execute",
    "execution",
    "overflow",
    "denial",
    "privilege",
    "bypass",
    "injection",
    "xss",
    "sql",
    "buffer",
    "memory",
    "authentication",
    "arbitrary",
    "root",
]

# Колонки CVSS-metric vector — НЕ используем как признаки (leakage)!
CVSS_COMPONENTS = [
    "access_authentication",
    "access_complexity",
    "access_vector",
    "impact_availability",
    "impact_confidentiality",
    "impact_integrity",
]


@dataclass
class SplitData:
    """Результат разбиения train/val/test."""
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


# --- Загрузка и очистка ---------------------------------------------------
def load_data(path=DATA_RAW) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Первая колонка без имени — это CVE-ID
    if df.columns[0] == "" or df.columns[0].startswith("Unnamed"):
        df = df.rename(columns={df.columns[0]: "cve_id"})
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Создаёт таргет severity по официальной шкале CVSS v2.

    Границы: LOW [0, 4), MEDIUM [4, 7), HIGH [7, 9), CRITICAL [9, 10].
    Левая граница включается (right=False), чтобы значения вроде 4.0
    попадали в MEDIUM, а не в LOW.
    """
    df = df.dropna(subset=["cvss", "summary"]).copy()
    df["severity"] = pd.cut(
        df["cvss"], bins=CVSS_BINS, labels=CVSS_LABELS, right=False,
    ).astype(int)
    return df


def _clip_outliers_iqr(series: pd.Series, k: float = 3.0) -> pd.Series:
    """Клиппинг по IQR (умеренный, k=3 чтобы не съесть валидную дисперсию)."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return series.clip(lower=lower, upper=upper)


def _add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    df["summary"] = df["summary"].fillna("").astype(str)
    df["desc_len"] = df["summary"].str.len()
    df["desc_word_count"] = df["summary"].str.split().str.len().fillna(0)
    df["desc_upper_ratio"] = df["summary"].apply(
        lambda s: sum(1 for c in s if c.isupper()) / max(len(s), 1)
    )
    df["desc_digit_ratio"] = df["summary"].apply(
        lambda s: sum(1 for c in s if c.isdigit()) / max(len(s), 1)
    )
    for word in KEYWORDS:
        df[f"has_{word}"] = df["summary"].str.contains(
            word, case=False, regex=False
        ).astype(int)
    return df


def _add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    pub = pd.to_datetime(df["pub_date"], errors="coerce")
    mod = pd.to_datetime(df["mod_date"], errors="coerce")
    df["year"] = pub.dt.year
    df["month"] = pub.dt.month
    df["days_to_mod"] = (mod - pub).dt.days.fillna(0).clip(lower=0)
    median_year = df["year"].median()
    df["year"] = df["year"].fillna(median_year).astype(int)
    df["month"] = df["month"].fillna(1).astype(int)
    return df


def _add_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    # CVSS-компоненты сохраняем для EDA, но НЕ используем как признаки.
    for col in CVSS_COMPONENTS:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN").astype(str)
    if "cwe_code" in df.columns:
        df["cwe_code"] = df["cwe_code"].fillna(-1).astype(int).astype(str)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Полная очистка + feature engineering."""
    initial = df.shape

    # 1. Дубли
    df = df.drop_duplicates()

    # 2. Таргет (одновременно убирает строки без cvss/summary)
    df = create_target(df)

    # 3. Текстовые фичи
    df = _add_text_features(df)

    # 4. Дата
    df = _add_date_features(df)

    # 5. Категориальные
    df = _add_categorical_features(df)

    # 6. Выбросы: cvss в [0, 10] по определению; desc_len и days_to_mod клипим.
    df = df[(df["cvss"] >= 0) & (df["cvss"] <= 10)].copy()
    df["desc_len"] = _clip_outliers_iqr(df["desc_len"])
    df["days_to_mod"] = _clip_outliers_iqr(df["days_to_mod"])

    # 7. Не утечь cvss в признаки: cvss используется только для построения
    # таргета, в обучающую матрицу он не идёт.
    print(f"[preprocess] {initial} -> {df.shape}")
    return df


# --- Разбиение -------------------------------------------------------------
def split_by_year(
    df: pd.DataFrame,
    val_year: int = 2017,
    test_year: int = 2019,
) -> SplitData:
    """Time-based split: train < val_year <= val < test_year <= test.

    Это защищает от data leakage: модель не видит будущих CVE на обучении.
    Альтернатива (random stratified) приводит к утечке, если в близких по
    времени CVE есть похожие формулировки.
    """
    feature_cols = (
        ["desc_len", "desc_word_count", "desc_upper_ratio", "desc_digit_ratio",
         "year", "month", "days_to_mod", "cwe_code"]
        + [c for c in df.columns if c.startswith("has_")]
        + ["summary"]
    )
    feature_cols = [c for c in feature_cols if c in df.columns]

    train_mask = df["year"] < val_year
    val_mask = (df["year"] >= val_year) & (df["year"] < test_year)
    test_mask = df["year"] >= test_year

    return SplitData(
        X_train=df.loc[train_mask, feature_cols].reset_index(drop=True),
        X_val=df.loc[val_mask, feature_cols].reset_index(drop=True),
        X_test=df.loc[test_mask, feature_cols].reset_index(drop=True),
        y_train=df.loc[train_mask, "severity"].reset_index(drop=True),
        y_val=df.loc[val_mask, "severity"].reset_index(drop=True),
        y_test=df.loc[test_mask, "severity"].reset_index(drop=True),
    )


def split_stratified(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = SEED,
) -> SplitData:
    """Стратифицированный random split — для сравнения с time-based."""
    from sklearn.model_selection import train_test_split

    feature_cols = [
        c for c in df.columns if c not in (
            "severity", "cvss", "cve_id", "pub_date", "mod_date", "cwe_name",
            *CVSS_COMPONENTS,
        )
    ]
    X = df[feature_cols]
    y = df["severity"]

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=val_size + test_size,
        stratify=y, random_state=seed,
    )
    rel = test_size / (val_size + test_size)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=rel, stratify=y_tmp, random_state=seed,
    )
    return SplitData(X_tr, X_val, X_te, y_tr, y_val, y_te)
