.PHONY: help install install-dev setup clean test lint format check run-api run-ui docker-up docker-down

help:
	@echo "Available commands:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  setup        - Run initial project setup"
	@echo "  clean        - Clean cache and build files"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black and isort"
	@echo "  check        - Run all checks (lint + type check)"
	@echo "  run-api      - Start FastAPI server"
	@echo "  run-ui       - Start Gradio interface"
	@echo "  docker-up    - Start Docker services"
	@echo "  docker-down  - Stop Docker services"

install:
	pip install -e .

install-dev:
	pip install -e .[dev]

setup:
	python scripts/setup.py

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

test:
	pytest -v

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

check: lint test

run-api:
	uvicorn powerbi_rag.api.main:app --reload --host 0.0.0.0 --port 8000

run-ui:
	python -m powerbi_rag.ui.gradio_app

docker-up:
	docker-compose --profile qdrant up -d

docker-down:
	docker-compose down