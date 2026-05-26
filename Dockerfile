# Multi-stage Dockerfile для CVE Severity Predictor
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Системные зависимости для lightgbm
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        make \
    && rm -rf /var/lib/apt/lists/*

# Кешируемый слой зависимостей
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Код
COPY src/ ./src/
COPY tests/ ./tests/
COPY Makefile pyproject.toml .flake8 ./

# Данные и модели подключаются как volume в compose
RUN mkdir -p data/raw data/processed models reports report/images

# Порт API
EXPOSE 8000

# По умолчанию запускаем обучение
CMD ["python", "-m", "src.train"]
