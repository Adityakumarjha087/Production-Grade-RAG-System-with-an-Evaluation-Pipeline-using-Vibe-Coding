# ragsys — hybrid RAG over regulatory PDFs

Production-style retrieval-augmented generation over SEBI circulars (or any PDF
corpus): layout-aware ingestion, hybrid retrieval (BM25 + dense → RRF → cross-encoder),
grounded answers with **validated citations**, calibrated **abstention**, an evaluation
harness that compares every configuration, and a streaming FastAPI service with
caching, cost accounting and Langfuse tracing.

Every design choice — and the measurement that drove it — is in **[DECISIONS.md](DECISIONS.md)**.

```
                ┌──────────── ingestion (offline job) ─────────────┐
 PDFs ──► Docling / PyMuPDF ──► Document (canonical text + element spans)
                                   │ chunk: fixed | recursive | structural
                                   ▼
                     bge-base (dense) + BM25 (sparse) ──► Qdrant  <alias>__<version>
                └───────────────────────────────────────────────────┘
                                                   alias swap (blue/green)
 query ─► cache? ─► dense ┐                          │
                   sparse ┴► RRF ─► cross-encoder ─► gate: all signals low? ─► "I don't know"
                                                     │ no
                                    Claude (stream) ─► validate [n] citations ─► answer
            every stage: latency · tokens · cost · Langfuse span · JSON log line
```

## Quick start

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/). No API key is needed for
the offline path.

```bash
uv sync --all-extras                    # deps incl. docling (ingest) and ragas (eval)
cp .env.example .env                    # defaults: local Qdrant on disk, configs/local.yaml

uv run ragsys sample-data               # 4 synthetic sample PDFs + 11 approved test items
uv run ragsys ingest --strategy all     # parse, chunk ×3, embed, index
uv run ragsys query "How long must intermediaries preserve KYC records?"
uv run ragsys eval                      # 3 chunkers × 5 retrieval variants -> report
uv run ragsys serve                     # http://localhost:8000/docs
```

`configs/local.yaml` uses the real embedding/reranker models but an **extractive,
deterministic answerer** and a lexical judge, so everything runs offline. To use Claude,
set `ANTHROPIC_API_KEY` in `.env` and `RAG_CONFIG=configs/base.yaml` (generation:
`claude-opus-5-5`; judge: `claude-sonnet-5-5`; test-set generation: `claude-opus-5-5`).

### With the real corpus

```bash
uv run ragsys fetch-sebi --limit 500    # polite, resumable download -> data/raw/sebi/
# (or copy your own PDFs into data/raw/)
uv run ragsys ingest --strategy all     # Docling parse is cached in data/parsed/
uv run ragsys testset from-specs configs/testsets/sebi_seed.yaml   # 28 hand-written items (no key needed)
uv run ragsys testset generate --n 150  # needs ANTHROPIC_API_KEY; items start "pending"
uv run ragsys testset review            # approve / edit / reject (resumable)
uv run ragsys eval                      # only approved items are scored
```

### Docker

```bash
uv run ragsys sample-data --config configs/local.yaml    # or put PDFs in data/raw
docker compose up -d --build                             # qdrant + redis + api
docker compose --profile jobs run --rm ingest            # build indexes inside Docker
./scripts/smoke_test.sh                                  # end-to-end checks
```

The serving image is ~760 MB (ONNX models, no torch). Build with
`--build-arg EXTRAS=ingest` to include Docling for in-container parsing.
`configs/docker.yaml` uses the PyMuPDF parser and the extractive answerer; switch
`generation.provider` to `anthropic` once a key is in `.env`.

## API

| Method | Path | |
|---|---|---|
| `POST` | `/v1/query` | `{"query": "..."}` → answer JSON |
| `POST` | `/v1/query/stream` | Server-Sent Events: `sources` → `token`* → `done` \| `error` |
| `GET` | `/healthz` | liveness |
| `GET` | `/readyz` | index present, cache reachable, models loaded |

```bash
curl -N localhost:8000/v1/query/stream -H 'content-type: application/json' \
     -d '{"query":"What is the deadline for pay-out of funds under T+1?"}'
```

Answer fields: `text`, `citations[]` (marker, doc title, pages, section, snippet),
`sources[]` (all context passages), `invalid_citations`, `abstained`,
`abstain_reason`, `confidence`, `model`, `usage`, `cost_usd`, `latency_ms`
(per stage, incl. time-to-first-token), `cached`, `request_id`, `trace_id`.
The `done` event's text is authoritative: if citation validation fails, it replaces the
streamed text with an abstention.

Every request emits one JSON log line (`event: rag.request`) with latency per stage,
token usage, cost, cache hit, abstention reason and the request/trace ids.

## Configuration

Behaviour lives in YAML, secrets in `.env`.

| What | Where |
|---|---|
| Parser, chunking strategy and sizes | `parsing`, `chunking` |
| Toggle each retrieval stage | `retrieval.dense/sparse/rerank.enabled`, `rrf_k`, `final_k` |
| Abstention thresholds | `generation.abstention` |
| LLM model, effort, refusal fallback | `generation` |
| Cache backend / TTL | `cache` |
| Prices per model | `pricing` |
| Experiment matrix | `configs/experiments/matrix.yaml` |

Configs compose with `extends: base.yaml`.

## Evaluation

* **Test set**: LLM-generated from sampled passages (factual / table / multi-hop /
  unanswerable). Each evidence quote must appear verbatim in the source or the item is
  discarded; then a human reviews it.
* **Gold labels are character spans**, not chunk ids, so one test set scores every
  chunking strategy fairly.
* **Retrieval**: recall@{1,3,5,10}, MRR, per-stage recall (dense / sparse / RRF / rerank),
  retrieval p50/p95 latency.
* **Answers**: RAGAS faithfulness + answer correctness, judged by a different model
  from the generator.
* **Abstention**: precision/recall on unanswerable items, false-abstention rate.
* Output: `eval/results/<timestamp>/report.md` (+ CSV, JSON, exact config, git SHA,
  test-set hash). LLM calls are disk-cached, so re-runs are free.

## Tests

```bash
make test-fast    # ~10 s, hermetic: no network, no keys, no downloads
make test-slow    # real ONNX embedder + both rerankers + Docling (downloads models once)
make check        # ruff + mypy --strict + all tests
```

210 tests cover:
* chunker contracts (exact spans, full coverage, stable ids)
* BM25 maths and identifier search
* the blue/green alias swap
* RRF
* abstention rules
* citation parsing
* the Claude adapter against the real SDK with a mocked HTTP transport
* cache (Redis via fakeredis, failure → miss)
* the API (JSON, SSE order, errors, request ids)
* the evaluation metrics and runner
* the full CLI workflow
* Langfuse OTLP export to a local endpoint

## Layout

```
src/ragsys/
  ingestion/   parsers/ (docling, pymupdf, shared builder) · chunkers/ · pipeline.py
  indexing/    bm25.py (tokenizer + sparse encoder) · qdrant_store.py (alias swap)
  embeddings/  fastembed + hashing embedder
  retrieval/   retrievers.py · fusion.py (RRF) · reranker.py · pipeline.py (toggles)
  generation/  llm/ (anthropic, extractive, disk cache) · prompts · citations · abstention · generator
  evaluation/  testset · review · retrieval_metrics · answer_metrics · runner · report
  api/         app (FastAPI + SSE) · middleware (request id) · schemas
  observability/ logging · tracing (Langfuse) · cost
  service.py   request orchestration      factory.py  composition root
  cache.py     exact-match cache          config.py   settings + YAML config
  cli.py       ragsys ingest|query|testset|eval|serve|sample-data
configs/       base.yaml · local.yaml · docker.yaml · experiments/matrix.yaml
tests/         unit/ (hermetic) · integration/ (CLI e2e, real models, Langfuse export)
```

## Observability

Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` (and `LANGFUSE_HOST`) to get one
trace per request: `rag.query` → `retrieval.dense|sparse|rrf|rerank` → `llm.generate`
(model, token usage, prompt, output). The `trace_id` is returned in every answer.
Use Langfuse Cloud, or self-host it with
[Langfuse's own compose file](https://langfuse.com/self-hosting). Self-hosting needs
Postgres, ClickHouse, Redis and S3, so it is deliberately not bundled here.

## Limitations

See the end of [DECISIONS.md](DECISIONS.md#known-limitations--next-steps). In
particular, quote answer-quality numbers only after running the RAGAS judge with a
real API key on your reviewed test set.

> The PDFs produced by `ragsys sample-data` are **synthetic and fictional**: test
> fixtures labelled as such on every page. They are not real regulatory documents.
