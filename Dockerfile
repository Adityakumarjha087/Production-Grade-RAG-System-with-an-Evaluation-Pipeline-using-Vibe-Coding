# syntax=docker/dockerfile:1.7
# Multi-stage build with uv. The default image is the slim *serving* image
# (fastembed/ONNX, no torch). Build with --build-arg EXTRAS=ingest to include
# Docling for layout-aware parsing (much larger: pulls torch).

FROM python:3.11-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
ARG EXTRAS=""
WORKDIR /app

# Dependencies first (cached layer), project second.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project ${EXTRAS:+--extra $EXTRAS}
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable ${EXTRAS:+--extra $EXTRAS}


FROM python:3.11-slim-bookworm AS runtime
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /models /app/data \
    && chown -R app:app /models /app
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=app:app configs ./configs
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FASTEMBED_CACHE_PATH=/models/fastembed \
    HF_HOME=/models/hf \
    RAG_CONFIG=configs/docker.yaml
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"
CMD ["ragsys", "serve", "--host", "0.0.0.0", "--port", "8000"]
