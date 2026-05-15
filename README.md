[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)

# ML Project — CVE Severity Predictor

**Студент:** Саушкин Николай Олегович
**Группа:** БИВ238

## Содержание
1. [Описание задачи](#описание-задачи)
2. [Источник данных](#источник-данных)
3. [Структура репозитория](#структура-репозитория)
4. [Запуск](#запуск)
5. [Docker](#docker)
6. [Качество кода](#качество-кода)
7. [Защита от data leakage](#защита-от-data-leakage)
8. [Результаты](#результаты)
9. [Отчёт](#отчёт)

## Описание задачи

Задача: многоклассовая классификация уязвимости по уровню критичности
(LOW / MEDIUM / HIGH / CRITICAL) на основе её текстового описания и
вспомогательных метаданных.

Целевая метрика: **F1-macro**. Класс CRITICAL — самый редкий и при этом
критичный с точки зрения практики, поэтому accuracy (которая поощряет
угадывание мажоритарного класса MEDIUM) не подходит. Accuracy приводится
как вторичная метрика, но решения принимаются по F1-macro.

## Источник данных

Датасет: [CVE Dataset (Kaggle, monetary)](https://www.kaggle.com/datasets/andrewkronser/cve-common-vulnerabilities-and-exposures).
Содержит ~89 660 CVE-записей за 1999–2019 годы с CVSS v2-метриками,
CWE-классификацией и текстовым описанием. Целевая колонка отсутствует
изначально — она построена дискретизацией CVSS score по границам
официальной шкалы CVSS v2 от FIRST.

## Структура репозитория

```
.
├── data/
│   ├── raw/                  # сырые данные (cve.csv) — не коммитим
│   └── processed/            # parquet после препроцессинга
├── models/                   # model.pkl, feature_pipeline.pkl, model_meta.json
├── notebooks/                # пусто
├── presentation/             # презентация для защиты
├── report/
│   ├── images/               # графики для отчёта
│   └── report.md             # финальный отчёт
├── reports/                  # артефакты: эксперименты, метрики, графики
│   ├── experiments.csv
│   ├── feature_importance.csv
│   ├── final_model_report.txt
│   ├── eda_findings.md
│   └── *.png
├── src/
│   ├── __init__.py
│   ├── config.py             # SEED, пути, маппинг severity
│   ├── preprocessing.py      # загрузка, очистка, feature engineering
│   ├── features.py           # ColumnTransformer (TF-IDF + num + cat)
│   ├── eda.py                # EDA + текстовые выводы
│   └── train.py              # обучение, эксперименты, HP search
├── tests/
│   ├── conftest.py
│   └── test_preprocessing.py
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── .flake8
├── .pre-commit-config.yaml
├── pyproject.toml            # black + isort + pytest конфиги
├── requirements.txt          # все версии зафиксированы
└── README.md
```

## Запуск

```bash
# 1. Клонировать репозиторий
git clone <url>
cd hseml-group-project-GafiTrue

# 2. Создать виртуальное окружение
python -m venv .venv
.venv/Scripts/activate            # Windows
# source .venv/bin/activate       # Linux/macOS

# 3. Установить зависимости
make install                       # или pip install -r requirements.txt

# 4. Положить cve.csv в data/raw/
#    скачать с Kaggle по ссылке выше
```

### EDA

```bash
make eda                           # генерирует графики + reports/eda_findings.md
```

### Обучение

```bash
make train                         # полный прогон (~30+ минут)
make train-quick                   # отладочный прогон на 8к строк (~5 минут)
```

Артефакты:
* `models/model.pkl` — финальная модель;
* `models/feature_pipeline.pkl` — ColumnTransformer для инференса;
* `models/model_meta.json` — имя модели + метрики на test;
* `reports/experiments.csv` — таблица всех экспериментов;
* `reports/feature_importance.csv` — топ-30 признаков на класс;
* `reports/final_model_report.txt` — classification report + confusion matrix.

## Docker

```bash
docker compose up --build train     # обучение
docker compose up --build eda       # EDA
docker compose up --build test      # тесты
```

`data/`, `models/`, `reports/`, `report/` подключены как volume, поэтому
артефакты остаются на хосте.

## Качество кода

```bash
make lint                          # flake8 + black --check + isort --check
make format                        # автоформат
make test                          # pytest
```

Pre-commit hooks устанавливаются командой `pre-commit install`. После
этого black/isort/flake8 запускаются автоматически на каждом коммите.

Воспроизводимость:
* глобальный `SEED = 42`, фиксируется `set_seed()` в `src/config.py` —
  устанавливает `PYTHONHASHSEED`, `random.seed`, `np.random.seed`;
* все версии зависимостей в `requirements.txt` закреплены;
* train/val/test разбиение детерминировано (time-based, не зависит
  от seed).

## Защита от data leakage

В исходном CSV есть колонки `access_authentication`, `access_complexity`,
`access_vector`, `impact_availability`, `impact_confidentiality`,
`impact_integrity`. Это компоненты CVSS v2 metric vector, **по которым
напрямую вычисляется CVSS score** (формула в спецификации FIRST). Поскольку
наш таргет `severity` получен дискретизацией CVSS, использование этих
колонок как признаков — прямая утечка таргета.

**Решение.** В `src/features.py` эти колонки исключены из feature pipeline.
Они хранятся в DataFrame только для EDA и для будущих сравнительных
экспериментов. Тест `test_feature_pipeline_excludes_cvss_components`
проверяет, что они не попадают в матрицу признаков.

**Дополнительно**: train/val/test разбиваются по году публикации
(`split_by_year`), а не случайно. Это вторая защита: модель не видит
будущие CVE на этапе обучения, что ближе к реальному сценарию использования.

## Результаты

Чекпоинт-снапшот (quick-режим, 8 000 строк, train=4185 / val=2538 /
test=1277). На полном датасете цифры выше; команда `make train` без
флагов воспроизводит.

| Модель | F1-macro (val) | F1-macro (test) | Accuracy (test) |
|---|---|---|---|
| LogReg baseline (text only) | 0.460 | 0.429 | 0.496 |
| LogReg (full features) | 0.422 | 0.367 | 0.383 |
| Naive Bayes | 0.442 | 0.393 | 0.494 |
| KNN (k=15) | 0.380 | 0.376 | 0.534 |
| **LinearSVC** | **0.490** | **0.455** | **0.590** |
| RandomForest | 0.364 | 0.295 | 0.635 |
| GradientBoosting | 0.379 | 0.311 | 0.636 |
| LightGBM | 0.501 | 0.449 | 0.629 |
| XGBoost | 0.404 | 0.334 | 0.637 |
| LogReg + SVD(100) | 0.383 | 0.329 | 0.326 |
| LogReg + SVD(300) | 0.415 | 0.360 | 0.363 |
| LogReg (tuned) | 0.396 | 0.342 | 0.336 |
| LightGBM (tuned) | 0.494 | 0.452 | 0.645 |
| Voting (soft) | 0.489 | 0.437 | 0.567 |
| Stacking | 0.469 | 0.422 | 0.643 |

Финальная модель: **LinearSVC** (f1_macro_test = 0.4553).
Полная таблица с гиперпараметрами и временем обучения —
в `reports/experiments.csv`.

## Отчёт

[`report/report.md`](report/report.md) — подробный отчёт по чекпоинт-критериям.
