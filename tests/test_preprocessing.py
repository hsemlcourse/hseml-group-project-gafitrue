"""Базовые тесты препроцессинга и feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import CVSS_BINS, SEED, set_seed
from src.features import build_feature_pipeline
from src.preprocessing import (
    CVSS_COMPONENTS,
    create_target,
    preprocess,
    split_by_year,
)


@pytest.fixture
def synthetic_df():
    """Маленький синтетический CVE-датасет."""
    set_seed()
    rng = np.random.default_rng(SEED)
    n = 200
    return pd.DataFrame(
        {
            "cve_id": [f"CVE-2020-{i:04d}" for i in range(n)],
            "mod_date": pd.date_range("2015-01-01", periods=n, freq="D"),
            "pub_date": pd.date_range("2015-01-01", periods=n, freq="D"),
            "cvss": rng.uniform(0, 10, size=n),
            "cwe_code": rng.integers(1, 100, size=n),
            "cwe_name": ["CWE-X"] * n,
            "summary": [
                (
                    "remote attacker can execute arbitrary code via buffer overflow"
                    if i % 2 == 0
                    else "denial of service via memory corruption"
                )
                for i in range(n)
            ],
            "access_authentication": ["NONE"] * n,
            "access_complexity": ["LOW"] * n,
            "access_vector": ["NETWORK"] * n,
            "impact_availability": ["PARTIAL"] * n,
            "impact_confidentiality": ["PARTIAL"] * n,
            "impact_integrity": ["PARTIAL"] * n,
        }
    )


def test_create_target_bins_match_cvss_v2(synthetic_df):
    """Границы дискретизации должны соответствовать спецификации CVSS v2."""
    df = create_target(synthetic_df.copy())
    assert df["severity"].between(0, 3).all()

    # Проверяем конкретные значения
    test_cases = [(3.9, 0), (4.0, 1), (6.9, 1), (7.0, 2), (8.9, 2), (9.0, 3), (10.0, 3)]
    for cvss, expected in test_cases:
        s = pd.Series([cvss])
        binned = pd.cut(s, bins=CVSS_BINS, labels=[0, 1, 2, 3], right=False).astype(int)
        assert binned.iloc[0] == expected, f"CVSS {cvss} -> {binned.iloc[0]}, expected {expected}"


def test_preprocess_creates_expected_features(synthetic_df):
    df = preprocess(synthetic_df.copy())
    expected = {
        "severity",
        "desc_len",
        "desc_word_count",
        "desc_upper_ratio",
        "desc_digit_ratio",
        "year",
        "month",
        "days_to_mod",
        "has_remote",
        "has_execute",
        "has_overflow",
    }
    assert expected.issubset(df.columns), f"missing: {expected - set(df.columns)}"


def test_preprocess_handles_outliers(synthetic_df):
    """Экстремальные desc_len должны быть клиппированы."""
    df = synthetic_df.copy()
    df.loc[0, "summary"] = "x" * 100000
    out = preprocess(df)
    # после клиппинга максимум должен быть значительно меньше 100k
    assert out["desc_len"].max() < 100000


def test_preprocess_drops_duplicates(synthetic_df):
    df = pd.concat([synthetic_df, synthetic_df], ignore_index=True)
    out = preprocess(df)
    assert len(out) == len(synthetic_df)


def test_time_based_split_has_no_overlap(synthetic_df):
    """train/val/test не должны пересекаться по годам."""
    df = preprocess(synthetic_df.copy())
    # подменяем годы вручную для теста
    df.loc[:60, "year"] = 2014
    df.loc[61:120, "year"] = 2017
    df.loc[121:, "year"] = 2019

    split = split_by_year(df, val_year=2017, test_year=2019)
    if len(split.X_train) > 0:
        assert split.X_train["year"].max() < 2017
    if len(split.X_val) > 0:
        assert split.X_val["year"].min() >= 2017
        assert split.X_val["year"].max() < 2019
    if len(split.X_test) > 0:
        assert split.X_test["year"].min() >= 2019


def test_feature_pipeline_excludes_cvss_components(synthetic_df):
    """Колонки CVSS-vector не должны попадать в матрицу признаков (leakage)."""
    df = preprocess(synthetic_df.copy())
    df.loc[:99, "year"] = 2014
    df.loc[100:, "year"] = 2019

    split = split_by_year(df, val_year=2017, test_year=2019)
    pipe = build_feature_pipeline()
    pipe.fit(split.X_train)
    feature_names = pipe.get_feature_names_out()

    for col in CVSS_COMPONENTS:
        leaked = [f for f in feature_names if col in f]
        assert not leaked, f"leakage: {col} found in features: {leaked[:3]}"


def test_feature_pipeline_is_deterministic(synthetic_df):
    """С зафиксированным seed два вызова дают одинаковую матрицу."""
    df = preprocess(synthetic_df.copy())
    df.loc[:99, "year"] = 2014
    df.loc[100:, "year"] = 2019
    split = split_by_year(df, val_year=2017, test_year=2019)

    set_seed()
    p1 = build_feature_pipeline()
    X1 = p1.fit_transform(split.X_train)

    set_seed()
    p2 = build_feature_pipeline()
    X2 = p2.fit_transform(split.X_train)

    assert X1.shape == X2.shape
    assert (X1 != X2).nnz == 0
