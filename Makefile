.PHONY: install format lint typecheck test check tree

install:
	uv sync --all-groups

format:
	uv run ruff format .

lint:
	uv run ruff check .

typecheck:
	uv run mypy src

test:
	uv run pytest --cov=src/footcast --cov-report=term-missing

check: format lint typecheck test

tree:
	find . -maxdepth 4 \
		-not -path "./.git/*" \
		-not -path "./.venv/*" \
		-not -path "./.ruff_cache/*" \
		-not -path "./.pytest_cache/*" \
		| sort
