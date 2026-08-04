.PHONY: install format lint typecheck test check tree \
	ingest-historical ingest-current ingest-all audit-bronze

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

ingest-historical:
	uv run python -m footcast.ingestion.historical \
		--start-year 2019 \
		--end-year 2025
ingest-current:
	uv run python -m footcast.ingestion.current

ingest-all: ingest-historical ingest-current audit-bronze

audit-bronze:
	uv run python -m footcast.ingestion.audit

tree:
	find . -maxdepth 4 \
		-not -path "./.git/*" \
		-not -path "./.venv/*" \
		-not -path "./.ruff_cache/*" \
		-not -path "./.pytest_cache/*" \
		| sort