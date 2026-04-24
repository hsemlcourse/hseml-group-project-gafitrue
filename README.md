[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)

# ML Project — CVE Severity Predictor

**Студент:** Саушкин Николай Олегович  

**Группа:** БИВ238  


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуск](#запуск)
4. [Данные](#данные)
5. [Результаты](#результаты)
6. [Отчёт](#отчёт)


## Описание задачи

Цель проекта — предсказать уровень критичности уязвимости (LOW / MEDIUM / HIGH / CRITICAL) на основе её текстового описания и метаданных.

**Задача:** Классификация  

**Датасет:** :contentReference[oaicite:0]{index=0}  

**Целевая метрика:** F1-macro (используется из-за несбалансированности классов)


## Структура репозитория

```

.
├── data
│   ├── processed               # (опционально) обработанные данные
│   └── raw                     # исходный датасет (cve.csv)
├── models                      # сохранённые модели (model.pkl, tfidf.pkl)
├── notebooks                   # (не используется на CP1)
├── presentation                # презентация для защиты
├── report
│   ├── images                  # графики и визуализации
│   └── report.md               # финальный отчёт
├── src
│   ├── preprocessing.py        # предобработка и feature engineering
│   ├── train.py               # обучение моделей и эксперименты
│   └── eda.py                 # визуализация данных
├── tests
│   └── test.py                # (опционально)
├── requirements.txt
└── README.md

````

---

## Запуск

```bash
# 1. Клонировать репозиторий
git clone <url>
cd <repo-name>

# 2. Создать виртуальное окружение
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 3. Установить зависимости
pip install -r requirements.txt
( pip install pandas numpy scikit-learn matplotlib seaborn joblib scipy)
````

### Запуск анализа данных (EDA)

```bash
python src/eda.py
```

Результат:

* `reports/severity_distribution.png`
* `reports/desc_length.png`
* `reports/cvss_distribution.png`

---

### Обучение модели

```bash
python src/train.py
```

Результат:

* `models/model.pkl`
* `models/tfidf.pkl`
* `reports/experiments.csv`

---

## Данные

* `data/raw/` — исходный файл `cve.csv`
* `data/processed/` — (необязательно)

### Особенность датасета

В исходных данных отсутствует целевая переменная severity.
Она была создана из CVSS score по следующей шкале:

* LOW: < 4
* MEDIUM: 4–6.9
* HIGH: 7–8.9
* CRITICAL: ≥ 9

---

## Результаты

| Модель                         | F1-macro | Accuracy | Примечание     |
| ------------------------------ | -------- | -------- | -------------- |
| Logistic Regression (baseline) | 0.55     | 0.58     | базовая модель |
| Naive Bayes                    | 0.42     | 0.54     | работает хуже  |
| RandomForest                   | **0.72** | **0.79** | лучшая модель  |

