.PHONY: install lint format test eda train train-quick clean docker-build docker-run all

PYTHON := python
PIP := pip

install:  ## Установить зависимости
	$(PIP) install -r requirements.txt

lint:  ## Запустить flake8 и проверку форматирования
	flake8 src tests
	black --check src tests
	isort --check-only src tests

format:  ## Автоформат: black + isort
	black src tests
	isort src tests

test:  ## Прогнать pytest
	$(PYTHON) -m pytest tests/ -v

eda:  ## EDA: визуализации + текстовые выводы
	$(PYTHON) -m src.eda

train:  ## Полное обучение всех моделей
	$(PYTHON) -m src.train

train-quick:  ## Быстрый прогон для отладки (8к строк)
	$(PYTHON) -m src.train --quick

clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache
	rm -rf models/*.pkl models/*.json reports/*.png reports/*.csv reports/*.md reports/*.txt

docker-build:
	docker build -t cve-severity-predictor .

docker-run:
	docker compose up --build

all: lint test eda train  ## Полная проверка + обучение
