.PHONY: \
	check \
	lint \
	format \
	typecheck \
	test \
	evaluate-baselines \
	train-logistic \
	train-hgb \
	train-ensemble \
	compare-day-4 \
	backtest-models \
	calibration-report \
	fit-production \
	predict-future \
	production \
	day-4\
	pipeline\
	pipeline-data\
	pipeline-models\
	pipeline-production\
	ci\
	validate-artifacts\
	docker-build \
	docker-build-dev \
	docker-health \
	docker-quality \
	docker-test \
	docker-models \
	docker-fit-production \
	docker-predict \
	docker-validate \
	docker-check
	

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

docker-build:
	docker build \
		--target runtime \
		--tag footcast:local \
		.

docker-build-dev:
	docker build \
		--target development \
		--tag footcast:development \
		.

docker-health: docker-build
	docker run \
		--rm \
		footcast:local \
		python scripts/container_healthcheck.py

docker-quality:
	docker compose run \
		--rm \
		quality

docker-test:
	docker compose run \
		--rm \
		test

docker-models:
	docker compose run \
		--rm \
		pipeline-models

docker-fit-production:
	docker compose run \
		--rm \
		fit-production

docker-predict:
	docker compose run \
		--rm \
		predict

docker-validate:
	docker compose run \
		--rm \
		validate-artifacts

docker-check: \
	docker-quality \
	docker-test \
	docker-health

production-check: fit-production validate-artifacts

validate-artifacts:
	uv run python scripts/validate_artifacts.py

ci:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy src
	uv run pytest \
		--cov=src/footcast \
		--cov-report=term-missing \
		--cov-report=xml:artifacts/coverage/coverage.xml \
		--cov-report=html:artifacts/coverage/html \
		--cov-fail-under=72

pipeline:
	uv run python -m footcast.pipelines.run_pipeline --pipeline full

pipeline-data:
	uv run python -m footcast.pipelines.run_pipeline --pipeline data

pipeline-models:
	uv run python -m footcast.pipelines.run_pipeline --pipeline models

pipeline-production:
	uv run python -m footcast.pipelines.run_pipeline --pipeline production

fit-production: train-ensemble
	uv run python -m footcast.prediction.fit_production

predict-future:
	uv run python -m footcast.prediction.predict

production: fit-production

train-ensemble: train-logistic train-hgb
	uv run python -m footcast.modelling.train_ensemble

compare-day-4:
	uv run python -m footcast.modelling.day_4_comparison

calibration-report:
	uv run python -m footcast.modelling.run_calibration_diagnostics

day-4: \
	backtest-models \
	calibration-report \
	train-ensemble \
	compare-day-4 \
	fit-production

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