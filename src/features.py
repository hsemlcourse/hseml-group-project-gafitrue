"""Сборка матрицы признаков через ColumnTransformer.

Используется один и тот же пайплайн на обучении и в API, чтобы избежать
расхождения признаков между train-time и serve-time.

Защита от data leakage
----------------------
В исходном CSV есть колонки access_vector, access_complexity,
access_authentication, impact_availability, impact_confidentiality,
impact_integrity — это компоненты CVSS v2 metric vector, из которых
**напрямую** вычисляется CVSS score. Поскольку наш таргет получен
дискретизацией CVSS, использование этих колонок как признаков — leakage.
В feature pipeline они НЕ участвуют. Берём только текст summary, число CWE
и метаданные публикации.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing import KEYWORDS


NUMERIC_FEATURES = [
    "desc_len",
    "desc_word_count",
    "desc_upper_ratio",
    "desc_digit_ratio",
    "year",
    "month",
    "days_to_mod",
] + [f"has_{w}" for w in KEYWORDS]

# cwe_code трактуем как категориальный (тысячи уникальных значений
# не дают осмысленного порядка)
CATEGORICAL_FEATURES = ["cwe_code"]


def build_feature_pipeline(
    tfidf_max_features: int = 5000,
    tfidf_ngram_range=(1, 2),
    tfidf_min_df: int = 3,
) -> ColumnTransformer:
    """Возвращает ColumnTransformer: text → TF-IDF, numeric → scale, cat → OHE."""
    text_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=tfidf_max_features,
            ngram_range=tfidf_ngram_range,
            min_df=tfidf_min_df,
            sublinear_tf=True,
            strip_accents="unicode",
        )),
    ])

    numeric_pipe = Pipeline([
        ("scale", StandardScaler(with_mean=False)),
    ])

    categorical_pipe = Pipeline([
        ("ohe", OneHotEncoder(
            handle_unknown="ignore", sparse_output=True, max_categories=200,
        )),
    ])

    return ColumnTransformer(
        transformers=[
            ("text", text_pipe, "summary"),
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )
