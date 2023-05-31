# Makefile
SHELL = /bin/bash

.PHONY: help
help:
	@echo "Commands:"
	@echo "poetry-env : creates a virtual environment."
	@echo "style : executes style formatting."
	@echo "test  : execute tests on code, data and models."

# Styling
.PHONY: style
style:
	black .
	flake8
	python3 -m isort .

# Environment
.ONESHELL:
poetry-env:
	poetry init
	poetry config virtualenvs.in-project true
	poetry shell
	poetry install
	pre-commit install && \
	pre-commit autoupdate

# Test
.PHONY: test
test:
	poetry run pytest -m "not training"

# .PHONY: dvc


.PHONY: model-run
model-run: 
	python3 main.py