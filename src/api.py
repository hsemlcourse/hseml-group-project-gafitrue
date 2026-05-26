"""FastAPI-сервис для CVE Severity Predictor.

Эндпоинты:
* ``GET  /``        — веб-интерфейс (одностраничный, ввод описания CVE);
* ``POST /predict``  — предсказание severity по JSON;
* ``GET  /health``   — проверка живости и факта загрузки модели;
* ``GET  /docs``     — авто-документация Swagger (даёт FastAPI).

Запуск локально::

    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

Перед запуском должны существовать ``models/model.pkl`` и
``models/feature_pipeline.pkl`` (см. ``make train`` / ``make train-quick``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.config import SEVERITY_MAP, set_seed
from src.inference import ModelNotTrainedError, predict

set_seed()

app = FastAPI(
    title="CVE Severity Predictor",
    description=(
        "Предсказание уровня критичности CVE-уязвимости "
        "(LOW / MEDIUM / HIGH / CRITICAL) по текстовому описанию."
    ),
    version="1.0.0",
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


# --- Схемы запроса/ответа -------------------------------------------------
class PredictRequest(BaseModel):
    summary: str = Field(
        ...,
        min_length=1,
        description="Текстовое описание уязвимости.",
    )
    cwe_code: Optional[int] = Field(
        None, description="Код CWE (например, 79 для XSS). Опционально."
    )
    pub_date: Optional[str] = Field(None, description="Дата публикации YYYY-MM-DD. Опционально.")
    mod_date: Optional[str] = Field(None, description="Дата модификации YYYY-MM-DD. Опционально.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": (
                        "Remote attacker can execute arbitrary code via a "
                        "crafted request due to a heap-based buffer overflow."
                    ),
                    "cwe_code": 787,
                }
            ]
        }
    }


class PredictResponse(BaseModel):
    severity_code: int
    severity: str
    confidence: float
    scores: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classes: list[str]


# --- Эндпоинты ------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    """Отдаёт веб-интерфейс."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    return "<h1>CVE Severity Predictor API</h1><p>См. /docs</p>"


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Проверка живости. ``model_loaded`` показывает, доступны ли артефакты."""
    try:
        from src.inference import _load_artifacts

        _load_artifacts()
        loaded = True
    except ModelNotTrainedError:
        loaded = False
    return HealthResponse(
        status="ok",
        model_loaded=loaded,
        classes=list(SEVERITY_MAP.values()),
    )


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest) -> PredictResponse:
    """Предсказывает severity по описанию CVE."""
    try:
        result = predict(
            summary=req.summary,
            cwe_code=req.cwe_code,
            pub_date=req.pub_date,
            mod_date=req.mod_date,
        )
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PredictResponse(**result)
