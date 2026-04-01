.PHONY: dev test lint format

## Full dev setup: install all packages + dev deps, install pre-commit hooks
dev:
	uv sync --all-packages --extra dev
	uv run pre-commit install

## Run the test suite
test:
	uv run pytest

## Lint check (no auto-fix)
lint:
	uv run ruff check

## Auto-format + auto-fix
format:
	uv run ruff check --fix
	uv run ruff format
