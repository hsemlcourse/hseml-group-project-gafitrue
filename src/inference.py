"""Инференс для API.

Один и тот же feature engineering, что и на обучении: входной текст
прогоняется через те же билдеры из ``src.preprocessing`` и тот же
сохранённый ``feature_pipeline.pkl``. Это гарантирует, что признаки на
serve-time совпадают с train-time (никакого ручного дублирования логики).

Финальная модель — LinearSVC. У неё нет ``predict_proba`` (только
``decision_function``), поэтому "уверенность" по классам мы получаем
softmax-нормализацией решающей функции. Это псевдо-вероятности для
отображения, а не калиброванные вероятности — см. замечание в отчёте
(раздел 6) про ``CalibratedClassifierCV``.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, SEVERITY_MAP
from src.preprocessing import (
    _add_categorical_features,
    _add_date_features,
    _add_text_features,
)

MODEL_PATH = MODELS_DIR / "model.pkl"
PIPELINE_PATH = MODELS_DIR / "feature_pipeline.pkl"


class ModelNotTrainedError(RuntimeError):
    """Бросается, если артефакты модели не найдены на диске."""


@lru_cache(maxsize=1)
def _load_artifacts():
    """Лениво загружает модель и feature pipeline (кешируется на процесс)."""
    if not MODEL_PATH.exists() or not PIPELINE_PATH.exists():
        raise ModelNotTrainedError(
            f"Не найдены артефакты модели в {MODELS_DIR}. "
            f"Сначала обучите модель: `make train` или `make train-quick`."
        )
    model = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)
    return model, pipeline


def _coerce_date(value: Optional[str], default: str) -> str:
    """Возвращает валидную дату-строку (YYYY-MM-DD).

    None, пустая строка или непарсящееся значение (например, плейсхолдер
    ``"string"`` из Swagger) заменяются на ``default``. Без этого одиночная
    невалидная дата давала NaN в году и роняла ``astype(int)`` в
    ``_add_date_features``.
    """
    if value is None or not str(value).strip():
        return default
    if pd.isna(pd.to_datetime(value, errors="coerce")):
        return default
    return value


def build_inference_frame(
    summary: str,
    cwe_code: Optional[int] = None,
    pub_date: Optional[str] = None,
    mod_date: Optional[str] = None,
) -> pd.DataFrame:
    """Собирает одну строку признаков ровно так же, как на обучении.

    Переиспользует ``_add_text_features`` / ``_add_date_features`` /
    ``_add_categorical_features`` из препроцессинга. Метаданные опциональны:
    невалидные/отсутствующие даты заменяются на сегодняшнюю; если CWE нет —
    служебное -1 (тот же placeholder, что и на обучении).
    """
    today = date.today().isoformat()
    pub = _coerce_date(pub_date, today)
    mod = _coerce_date(mod_date, pub)
    df = pd.DataFrame(
        {
            "summary": [summary if summary is not None else ""],
            "pub_date": [pub],
            "mod_date": [mod],
            "cwe_code": [cwe_code if cwe_code is not None else -1],
        }
    )
    df = _add_text_features(df)
    df = _add_date_features(df)
    df = _add_categorical_features(df)
    return df


def _scores_to_confidence(model, X) -> np.ndarray:
    """Возвращает псевдо-вероятности по 4 классам (сумма = 1).

    predict_proba -> как есть; decision_function -> softmax; иначе -> uniform.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[0]
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X)).reshape(1, -1)[0]
        scores = scores - scores.max()  # численная стабильность
        exp = np.exp(scores)
        return exp / exp.sum()
    n = len(SEVERITY_MAP)
    return np.full(n, 1.0 / n)


def predict(
    summary: str,
    cwe_code: Optional[int] = None,
    pub_date: Optional[str] = None,
    mod_date: Optional[str] = None,
) -> dict:
    """Предсказывает severity по описанию CVE.

    Возвращает: предсказанный класс (код + метка) и распределение
    уверенности по всем четырём классам.
    """
    if not summary or not summary.strip():
        raise ValueError("summary не должен быть пустым")

    model, pipeline = _load_artifacts()
    frame = build_inference_frame(summary, cwe_code, pub_date, mod_date)
    X = pipeline.transform(frame)

    pred_idx = int(model.predict(X)[0])
    confidence = _scores_to_confidence(model, X)

    scores = {SEVERITY_MAP[i]: round(float(confidence[i]), 4) for i in range(len(SEVERITY_MAP))}
    return {
        "severity_code": pred_idx,
        "severity": SEVERITY_MAP[pred_idx],
        "confidence": round(float(confidence[pred_idx]), 4),
        "scores": scores,
    }
