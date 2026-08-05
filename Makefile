.PHONY: install format lint typecheck test check tree \
	ingest-historical ingest-current ingest-all audit-bronze \
	build-silver build-form-features build-elo-features build-data
	build-model-dataset audit-model-dataset inspect-model-data evaluate-baselines
	train-logistic train-hgb compare-models day-3 backtest-models day-4 calibration-report
	train-ensemble compare-day-4

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

train-ensemble: train-logistic train-hgb
	uv run python -m footcast.modelling.train_ensemble

compare-day-4:
	uv run python -m footcast.modelling.day_4_comparison

calibration-report:
	uv run python -m footcast.modelling.run_calibration_diagnostics

day-4: backtest-models calibration-report train-ensemble compare-day-4

backtest-models:
	uv run python -m footcast.modelling.run_backtests

train-hgb:
	uv run python -m footcast.modelling.train_gradient_boosting

compare-models:
	uv run python -m footcast.modelling.model_comparison

day-3: evaluate-baselines train-logistic train-hgb compare-models

train-logistic:
	uv run python -m footcast.modelling.train_logistic

evaluate-baselines:
	uv run python -m footcast.modelling.run_baselines

inspect-model-data:
	uv run python -c 'from pathlib import Path; from footcast.modelling.dataset import build_model_datasets, load_model_dataset; df = load_model_dataset(Path("data/gold/model_dataset.parquet")); ds = build_model_datasets(df); print("train", ds.train.features.shape); print("validation", ds.validation.features.shape); print("test", ds.test.features.shape)'

build-model-dataset:
	uv run python -m footcast.features.model_dataset \
		--validation-season 2023/24 \
		--test-season 2024/25

audit-model-dataset:
	uv run python - <<'PY'
	import polars as pl
	from footcast.features.model_validation import validate_model_dataset

	df = pl.read_parquet("data/gold/model_dataset.parquet")
	validate_model_dataset(df)

	print("MODEL DATASET AUDIT")
	print("=" * 40)
	print("Rows:", df.height)
	print("Columns:", df.width)
	print("Unique matches:", df["match_key"].n_unique())
	print()
	print(df.group_by("split").len().sort("split"))
	print()
	print(
	    df.filter(pl.col("target").is_not_null())
	    .group_by(["split", "target"])
	    .len()
	    .sort(["split", "target"])
	)
	PY

build-elo-features:
	uv run python -m footcast.features.build_elo_features

build-data: build-silver build-form-features \
	build-elo-features build-model-dataset

build-form-features:
	uv run python -m footcast.features.build_features

build-silver:
	uv run python -m footcast.processing.clean_matches

ingest-historical:
	uv run python -m footcast.ingestion.historical \
		--start-year 2019 \
		--end-year 2025
ingest-current:
	uv run python -m footcast.ingestion.current

ingest-all: ingest-historical ingest-current audit-bronze

audit-bronze:
	uv run python -m footcast.ingestion.audit

build-data: build-silver build-form-features

tree:
	find . -maxdepth 4 \
		-not -path "./.git/*" \
		-not -path "./.venv/*" \
		-not -path "./.ruff_cache/*" \
		-not -path "./.pytest_cache/*" \
		| sort