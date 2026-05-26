"""Тесты инференса и FastAPI-эндпоинтов.

Не зависят от наличия обученной модели на диске: фикстура обучает крошечный
LinearSVC на синтетике и подменяет загрузчик артефактов через monkeypatch.
``build_inference_frame`` тестируется как чистая функция (модель не нужна).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.svm import LinearSVC

from src import inference
from src.api import app
from src.config import SEED, set_seed
from src.features import NUMERIC_FEATURES, build_feature_pipeline
from src.inference import ModelNotTrainedError, build_inference_frame


# --- build_inference_frame (без модели) -----------------------------------
def test_inference_frame_has_all_feature_columns():
    frame = build_inference_frame("remote code execution via buffer overflow")
    # текстовые + числовые + keyword-флаги + cwe_code
    for col in NUMERIC_FEATURES + ["summary", "cwe_code"]:
        assert col in frame.columns, f"missing feature column: {col}"
    assert len(frame) == 1


def test_inference_frame_handles_missing_metadata():
    """Без дат и CWE строка всё равно собирается (подставляются дефолты)."""
    frame = build_inference_frame("denial of service")
    assert frame["cwe_code"].iloc[0] == "-1"
    assert frame["year"].iloc[0] > 0
    assert frame["days_to_mod"].iloc[0] == 0


def test_inference_frame_keyword_flags():
    frame = build_inference_frame("remote attacker can execute arbitrary code")
    assert frame["has_remote"].iloc[0] == 1
    assert frame["has_execute"].iloc[0] == 1
    assert frame["has_xss"].iloc[0] == 0


def test_inference_frame_invalid_date_string():
    """Непарсящаяся дата (плейсхолдер 'string' из Swagger) не должна ронять."""
    frame = build_inference_frame("buffer overflow", pub_date="string", mod_date="string")
    assert int(frame["year"].iloc[0]) > 0
    assert int(frame["month"].iloc[0]) >= 1
    assert frame["days_to_mod"].iloc[0] == 0


# --- Фикстура: крошечная обученная модель ---------------------------------
@pytest.fixture
def trained_artifacts(monkeypatch):
    set_seed()
    rng = np.random.default_rng(SEED)
    n = 240
    summaries = []
    labels = []
    for i in range(n):
        cls = i % 4
        labels.append(cls)
        summaries.append(
            {
                0: "minor information exposure in error message",
                1: "cross-site scripting via the search field",
                2: "buffer overflow allows code execution",
                3: "unauthenticated remote root via command injection",
            }[cls]
        )
    df = pd.DataFrame(
        {
            "summary": summaries,
            "pub_date": pd.date_range("2016-01-01", periods=n, freq="D"),
            "mod_date": pd.date_range("2016-01-02", periods=n, freq="D"),
            "cwe_code": rng.integers(1, 100, size=n).astype(str),
        }
    )
    from src.preprocessing import (
        _add_categorical_features,
        _add_date_features,
        _add_text_features,
    )

    df = _add_text_features(df)
    df = _add_date_features(df)
    df = _add_categorical_features(df)
    y = pd.Series(labels)

    pipe = build_feature_pipeline(tfidf_max_features=200, tfidf_min_df=1)
    X = pipe.fit_transform(df)
    model = LinearSVC(class_weight="balanced", random_state=SEED, max_iter=2000)
    model.fit(X, y)

    monkeypatch.setattr(inference, "_load_artifacts", lambda: (model, pipe))
    return model, pipe


# --- predict() ------------------------------------------------------------
def test_predict_returns_valid_severity(trained_artifacts):
    result = inference.predict("remote attacker executes arbitrary code")
    assert result["severity"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert 0 <= result["confidence"] <= 1
    # scores — псевдо-вероятности, округлены до 4 знаков -> сумма ≈ 1
    assert abs(sum(result["scores"].values()) - 1.0) < 1e-3


def test_predict_empty_raises(trained_artifacts):
    with pytest.raises(ValueError):
        inference.predict("   ")


# --- FastAPI эндпоинты -----------------------------------------------------
def test_health_endpoint():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["classes"] == ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def test_index_served():
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "CVE" in r.text


def test_predict_endpoint(trained_artifacts):
    client = TestClient(app)
    r = client.post(
        "/predict",
        json={
            "summary": "unauthenticated remote root via command injection",
            "cwe_code": 78,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["severity"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert set(body["scores"]) == {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def test_predict_endpoint_swagger_default_body(trained_artifacts):
    """Дефолтное тело из Swagger (с 'string' в датах) должно давать 200."""
    client = TestClient(app)
    r = client.post(
        "/predict",
        json={
            "summary": "remote attacker executes arbitrary code",
            "cwe_code": 0,
            "pub_date": "string",
            "mod_date": "string",
        },
    )
    assert r.status_code == 200
    assert r.json()["severity"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def test_predict_endpoint_validation_error():
    """Пустой summary отсекается схемой Pydantic (422)."""
    client = TestClient(app)
    r = client.post("/predict", json={"summary": ""})
    assert r.status_code == 422


def test_predict_endpoint_no_model(monkeypatch):
    """Если модель не обучена — 503 с понятным сообщением."""

    def _raise():
        raise ModelNotTrainedError("no model")

    monkeypatch.setattr(inference, "_load_artifacts", _raise)
    client = TestClient(app)
    r = client.post("/predict", json={"summary": "buffer overflow"})
    assert r.status_code == 503
