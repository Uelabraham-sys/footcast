# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.11.32
ARG BUILD_VERSION=development
ARG BUILD_REVISION=unknown
ARG BUILD_CREATED=unknown

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-source

FROM python:${PYTHON_VERSION}-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:${PATH}"

COPY --from=uv-source /uv /uvx /usr/local/bin/

WORKDIR /app

RUN groupadd \
        --gid 10001 \
        footcast \
    && useradd \
        --uid 10001 \
        --gid footcast \
        --create-home \
        --shell /usr/sbin/nologin \
        footcast \
    && mkdir -p \
        /app/data \
        /app/artifacts \
    && chown -R \
        footcast:footcast \
        /app


FROM base AS dependencies

COPY pyproject.toml uv.lock README.md ./

RUN uv sync \
    --locked \
    --no-dev \
    --no-install-project

FROM dependencies AS runtime

ARG BUILD_VERSION
ARG BUILD_REVISION
ARG BUILD_CREATED

LABEL org.opencontainers.image.title="FootCast" \
      org.opencontainers.image.description="Football data engineering and probabilistic forecasting pipeline" \
      org.opencontainers.image.version="${BUILD_VERSION}" \
      org.opencontainers.image.revision="${BUILD_REVISION}" \
      org.opencontainers.image.created="${BUILD_CREATED}" \
      org.opencontainers.image.source="https://github.com/Uelabraham-sys/footcast"


COPY --chown=footcast:footcast src ./src
COPY --chown=footcast:footcast scripts ./scripts

RUN uv sync \
    --locked \
    --no-dev \
    --no-editable \
    && rm -rf /root/.cache/uv

RUN chown -R \
    footcast:footcast \
    /app/.venv \
    /app/data \
    /app/artifacts

USER footcast

HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=5s \
    --retries=3 \
    CMD ["python", "scripts/container_healthcheck.py"]

CMD ["python", "-m", "footcast.prediction.predict"]


FROM base AS development-dependencies

COPY pyproject.toml uv.lock README.md ./

RUN uv sync \
    --locked \
    --all-groups \
    --no-install-project


FROM development-dependencies AS development

COPY --chown=footcast:footcast src ./src
COPY --chown=footcast:footcast scripts ./scripts
COPY --chown=footcast:footcast tests ./tests
COPY --chown=footcast:footcast pyproject.toml uv.lock README.md Makefile ./

RUN uv sync \
    --locked \
    --all-groups \
    && rm -rf /root/.cache/uv

RUN chown -R \
    footcast:footcast \
    /app/.venv \
    /app/data \
    /app/artifacts

USER footcast

CMD ["pytest"]