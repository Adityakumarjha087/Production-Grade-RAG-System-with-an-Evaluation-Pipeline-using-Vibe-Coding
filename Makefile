.PHONY: install lint typecheck test test-fast test-slow check sample-data ingest eval serve up down smoke

install:            ## install all deps (incl. docling + ragas extras) into .venv
	uv sync --all-extras

lint:
	uv run ruff check src tests && uv run ruff format --check src tests

typecheck:
	uv run mypy

test-fast:          ## hermetic unit + CLI tests (~10s, no network)
	uv run pytest -m "not slow"

test-slow:          ## real ONNX models + Docling (downloads ~1GB on first run)
	uv run pytest -m slow

test:               ## everything
	uv run pytest

check: lint typecheck test

sample-data:        ## synthetic sample PDFs + approved test set
	uv run ragsys sample-data

ingest:             ## build indexes for all chunking strategies
	uv run ragsys ingest --strategy all

eval:               ## run the experiment matrix -> eval/results/<ts>/report.md
	uv run ragsys eval

serve:
	uv run ragsys serve --reload

up:                 ## qdrant + redis + api in Docker
	docker compose up -d --build

down:
	docker compose down

smoke:              ## end-to-end check against the running Docker stack
	./scripts/smoke_test.sh
