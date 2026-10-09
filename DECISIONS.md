# Design decisions

Each entry: **context → decision → alternatives considered → consequences**.
Numbers quoted here were measured during development (M1 Pro, 16 GB) on the
synthetic sample corpus unless stated otherwise; re-measure on the real corpus.

---

## D-001 · No LLM framework in the request path

**Context.** LangChain/LlamaIndex make a first demo fast but hide prompts, retries and
data flow behind abstractions that change between minor versions.
**Decision.** Plain Python modules behind small `Protocol`s (`Parser`, `Chunker`,
`Embedder`, `Reranker`, `LLMClient`, `ResponseCache`, `Tracer`). One composition root
(`factory.py`) turns config into objects.
**Alternatives.** LangChain LCEL; LlamaIndex query engines.
**Consequences.** More code to write, but every prompt and every stage is visible,
unit-testable and swappable in tests. LangChain appears only inside the *evaluation*
extra, because RAGAS needs it (D-014).

## D-002 · Docling as primary parser, PyMuPDF as fallback, parse once

**Context.** Regulatory circulars contain numbered clauses, multi-level headings and
tables (fee slabs, timelines). Structure-aware chunking needs a real heading
hierarchy and clean table cells.
**Decision.** Docling (layout model + table-structure model) by default; PyMuPDF
(font-size heading heuristics + `find_tables`) as automatic fallback when Docling
fails or returns empty text. Parsed documents are cached as JSON
(`data/parsed/<parser>/<doc_id>.json`), keyed by a content hash of the PDF.
**Alternatives.** `unstructured` (heavier, weaker tables in tests), pdfplumber,
cloud document AI (cost, data residency).
**Consequences.** Docling is slow on CPU and pulls in torch → it is an optional
`ingest` extra and not in the serving image (D-024). Parsing cost is paid once;
re-chunking for experiments never re-parses. Two PyMuPDF pitfalls found and fixed:
`Table.to_markdown()` invented an "external header" from the text above the table
and HTML-escaped hyphens (`pay&amp;#45;in`), so we render markdown from
`table.extract()` ourselves; wrapped titles became two headings, so adjacent
same-level headings are merged. Docling drops list markers (`3.1`) — we re-attach
them because clause numbers are what users cite.

## D-003 · Canonical document text with element offsets

**Context.** Citations need page/section; evaluation needs a way to say "this chunk
contains the answer" that does not depend on how the document was chunked.
**Decision.** Every parser emits flat `RawElement`s; one shared builder joins them
into `Document.text` and records `char_start/char_end`, page and heading breadcrumb
for each element. Every chunk stores its source span `[char_start, char_end)`.
**Consequences.** Page numbers and sections for citations come for free for every
chunker; gold evidence can be stored as spans (D-015). Invariant enforced in tests:
`doc.text[e.char_start:e.char_end] == e.text`.

## D-004 · Three chunkers behind one interface

| Strategy | Rule | Purpose |
|---|---|---|
| `fixed` | word-aligned character windows with overlap | control group |
| `recursive` | split on `\n\n` > `\n` > `. ` > `; ` > ` `, then greedy merge with overlap | standard baseline |
| `structural` | never split an element; start a new chunk at section boundaries (unless the chunk is still below `min_chunk_size`); oversized tables split by row groups **with the header row repeated**; oversized paragraphs fall back to recursive; chunk text prefixed with `Title > Section` breadcrumb | the hypothesis |

**Consequences.** All three are pure functions of a `Document` with deterministic
ids `sha256(doc_id|strategy|params|ordinal)` → re-ingestion is an upsert. Contract
tests (spans valid, every word covered, ids stable) run against all three.

## D-005 · Chunk size measured in characters, not tokens

**Decision.** `chunk_size=1200` chars (~300 tokens), well under bge's 512-token limit.
**Alternatives.** Tokenizer-exact sizing.
**Consequences.** Tokenizer-agnostic and deterministic across embedding models; the
cost is a looser bound (1 token ≈ 4 chars for English prose; tables are denser).

## D-006 · BM25 as Qdrant sparse vectors, custom tokenizer

**Context.** Hybrid search needs a lexical index. Users search for exact identifiers
(`SEBI/HO/MRD/2023/12`, `Regulation 23(1)(b)`) where embeddings are weak.
**Decision.** Store BM25 in the same Qdrant point as the dense vector: we compute the
TF-saturation half `tf·(k1+1)/(tf + k1·(1−b+b·|d|/avgdl))` client-side, Qdrant's
`Modifier.IDF` supplies IDF, query terms weigh 1 → dot product = BM25. Tokenizer keeps
compound identifiers and clause references whole *and* emits their parts; stopwords
removed; Snowball stemming. `avgdl` is computed at index time and stored in
collection metadata.
**Alternatives.** `rank_bm25`/`bm25s` in-process (a pickled index the API must load
and keep in sync), OpenSearch (another service), fastembed's BM25 (no control over
tokenization).
**Consequences.** One store, stateless API replicas. A test proves an exact circular
number retrieves the right document. Limitation: `avgdl` is per build; incremental
appends would use a stale value (we always rebuild — D-010).

## D-007 · Reciprocal Rank Fusion, client-side

**Decision.** `score(d) = Σ 1/(k + rank)`, `k = 60`, computed in Python.
**Why RRF.** Uses ranks only, so BM25 (unbounded) and cosine (bounded) need no score
normalisation; rewards agreement between retrievers.
**Why client-side** (Qdrant can fuse server-side): each stage's output is kept, traced
and evaluated separately — the report shows recall@k *per stage*, which is how we
prove fusion and reranking add value. Cost: one extra round-trip.

## D-008 · Embeddings: `BAAI/bge-base-en-v1.5` via fastembed (ONNX)

**Decision.** Local ONNX model, no torch. Query instruction prefix handled by
fastembed's `query_embed`.
**Alternatives.** OpenAI/Voyage embeddings (cost per re-index, network, non-reproducible
evals); bge-small (faster, lower quality).
**Consequences.** Free, offline, reproducible, slim image. Swappable via config; the
collection name includes the model slug so two models can coexist.

## D-009 · Reranker: MiniLM-L12 by default, bge-reranker-base as an eval variant

**Context.** First Docker run with `bge-reranker-base` (278M params): rerank stage
**5.3 s** of a 5.6 s request. Benchmark, 14 passages × ~500 tokens:

| model | host (M1, CoreML available) | Docker Desktop VM (CPU EP) |
|---|---:|---:|
| BAAI/bge-reranker-base | 1994 ms | 5913 ms |
| Xenova/ms-marco-MiniLM-L-12-v2 | 499 ms | 1794 ms |

**Decision.** Default `ms-marco-MiniLM-L-12-v2` with 20 candidates; keep
`hybrid_rerank_bge` in the evaluation matrix so the quality/latency trade-off is
decided by data on the real corpus, not by assumption. The eval runner caches one
reranker *per model* (a bug where the first model was reused for all variants was
caught by adding this variant).
**Consequences.** ~4× faster reranking. Docker Desktop on macOS is ~3× slower than
native; Linux servers or a GPU change the numbers again — re-measure where you deploy.

## D-010 · Versioned collections + atomic alias swap (blue/green indexing)

**Decision.** Index version = hash(chunk ids + chunk texts + embedding model + BM25
params). Build into `<alias>__<version>`, then atomically repoint the alias; delete
only the collection the alias previously pointed to. Same version already present →
no-op (idempotent ingestion).
**Alternatives.** Recreate in place (downtime, half-built index visible); delete all
other versions (would delete a concurrent job's in-progress build — the first
implementation did this; a test caught it).
**Consequences.** Zero-downtime re-index; the version doubles as a cache-key component
(D-018). Orphans from crashed builds are not auto-collected (documented limitation).

## D-011 · Citations are validated by code, not trusted

**Decision.** Passages are numbered `[1]..[n]` inside `<context>` tags; the model must
cite after each claim. `citations.py` parses `[1]`, `[1, 2]`, `[2-4]`, keeps markers
that refer to a passage in the prompt, and reports the rest as
`invalid_citations` (returned to clients, logged, traced).
**Consequences.** Hallucinated citations are measurable instead of silent. Citations
carry doc title, page range, section path and a snippet.

## D-012 · Three-layer abstention; the retrieval gate requires *all* signals to agree

**Layers.** (1) pre-generation gate on calibrated retrieval signals — skips the LLM
entirely; (2) instructed abstention sentence detected in the output; (3) an answer
with zero valid citations is replaced by an abstention.
**Evidence that shaped layer 1** (real models, sample corpus):

| question | rerank (bge) | dense cosine |
|---|---:|---:|
| off-topic ("capital of France") | 0.0009 | 0.31 |
| heavy paraphrase ("cost cap for passive funds like ETFs") | **0.0026** | **0.71** |
| near-topic, unanswerable ("max TER for *debt* schemes") | **0.87** | 0.79 |

A single cross-encoder threshold would have refused the paraphrase (false
abstention). **Decision:** abstain pre-generation only if *every* available signal
(cross-encoder top score, max dense cosine) is below its threshold; BM25-only configs
have no calibrated signal and defer to the LLM. The third row shows why layers 2–3
exist: no retrieval score can tell "on topic" from "answers the question".
**Calibration.** Thresholds live in config; the eval report's `abstain_recall` and
`false_abstain_rate` on the unanswerable/answerable items are the tuning signal.

## D-013 · Generator: Claude Opus 5.5 at `effort: low`, server-side fallback, streamed

**Decision.** Official `anthropic` SDK, streaming for every call
(`messages.stream` + `get_final_message()` → no HTTP timeouts, exact usage).
`output_config.effort=low` — grounded extraction over ~6 short passages doesn't need
deep reasoning. Server-side refusal fallback (`fallbacks="default"`, beta
`server-side-fallback-2026-07-01`) is on: if a safety classifier declines, the API
retries on the recommended fallback model in the same call. Cost is computed from the
model that actually served (`message.model`), prices from config.
**Consequences.** Refusals are surfaced as `abstain_reason=provider_refusal` and never
cached. The adapter is tested against the real SDK with a mocked HTTP transport
(request body, SSE parsing, usage, refusal, 429 mapping).

## D-014 · LLM-as-judge: RAGAS with a *different* model; no forced tool calls

**Decision.** Faithfulness and correctness via RAGAS (`Faithfulness`,
`AnswerCorrectness(weights=[1, 0])` — purely LLM-judged, no embedding similarity term)
with **Claude Sonnet 5.5 as judge** while Opus 5.5 generates (avoids
self-preference bias). RAGAS's LangChain wrapper is used with
`bypass_temperature/bypass_n` because current Claude models reject non-default
sampling parameters. RAGAS's newer instructor-based path was avoided: it forces
`tool_choice`, which these models reject.
**Pinning.** `ragas 0.4.3` breaks with `langchain-community ≥ 0.4.2` (import error) —
pinned `<0.4.2` in the `eval` extra.
**Offline proxy.** `judge_backend: lexical` (token support/F1) for CI and keyless runs —
explicitly *not* a substitute for an LLM judge.

## D-015 · Gold labels are character spans, not chunk ids

**Context.** Comparing chunkers with chunk-id labels is impossible: ids differ per
strategy.
**Decision.** Evidence = `(doc_id, char_start, char_end, text)` in the canonical
document text. A chunk is relevant iff same doc and
`overlap / min(|chunk|, |span|) ≥ 0.5`. recall@k = fraction of an item's spans hit in
the top k (multi-hop items have several); MRR over the first relevant chunk.
**Robustness.** If the corpus is re-parsed (other parser/version), spans are
re-anchored by their stored text; items that can't be re-anchored are dropped and
reported rather than scored against wrong offsets.

## D-016 · Test-set generation with verification and human review

**Decision.** Passages are sampled round-robin across documents (coverage), tables
separately. Question mix: 50 % factual, 20 % table, 15 % multi-hop (two passages),
15 % unanswerable. Structured JSON output; each evidence quote **must be found
verbatim** in its passage or the item is discarded. Every item starts `pending`;
`ragsys testset review` approves/edits/rejects, saving after each decision; only
`approved` items are evaluated.

## D-017 · Evaluation cost control and reproducibility

Indexes built once per chunker and shared across retrieval variants; models loaded
once; LLM calls memoised on disk by request hash (re-runs are free); answer metrics
on a fixed seeded subset; `--no-answers` for retrieval-only sweeps. Each run writes
`report.md`, `results.csv`, `results.json` and the exact `config.yaml`, plus git SHA
and a test-set content hash.

## D-018 · Exact-match response cache in Redis

**Key** = sha256(normalised query | config fingerprint | index version). The
fingerprint hashes behaviour-relevant config only (not paths/pricing), so a config
change or a re-index can never serve a stale answer.
**Rejected:** semantic caching — in a regulatory domain "similar" questions can have
different correct answers. Redis errors degrade to a cache miss; refusals aren't
cached; Redis runs as a bounded LRU with persistence off (it's a cache).

## D-019 · API: SSE streaming with a typed event protocol

`sources` (context passages, before any token) → `token`* → `done` (validated answer,
usage, cost, latency, trace id) or `error`. The `done` text is authoritative (it may
replace the streamed text with an abstention after citation validation). Request-ID
middleware is pure ASGI (no response buffering), propagates `X-Request-ID`, and binds
it to every log line. Upstream rate limits map to 503, other LLM errors to 502;
internals never leak into error bodies.

## D-020 · Sync retrieval in a thread pool, async generation

Embedding and cross-encoding are CPU-bound, so an async Qdrant client would not help;
retrieval runs via `anyio.to_thread` to keep the event loop free while LLM tokens
stream asynchronously. One code path serves API, CLI and evaluation.

## D-021 · Tracing with explicit span handles; tracing can't break requests

Spans are passed explicitly (`parent.child(...)`) instead of relying on OpenTelemetry
current-context, which doesn't survive an async generator resumed by Starlette in a
different task (SSE). Every Langfuse call is wrapped; failures log a warning. Without
keys a no-op tracer is used. Trace tree: `rag.query` → `retrieval.{dense,sparse,rrf,rerank}`
→ `llm.generate` (generation with model, usage, cost metadata). Verified by exporting
through the real SDK to a local OTLP endpoint in tests. Langfuse v4 deprecates
trace-level input/output — the root observation carries it.
**Hosting:** Langfuse Cloud recommended; self-hosted v3+ needs Postgres, ClickHouse,
Redis and S3 — not bundled here to keep the stack small.

## D-022 · Config: behaviour in YAML, secrets in env

`configs/base.yaml` + overlays via `extends:` (`local.yaml`, `docker.yaml`) and the
experiment matrix (deep-merged overlays). `.env` holds only keys/endpoints. Pricing is
config, not code.

## D-023 · Offline implementations of every external dependency

`HashEmbedder`, `LexicalOverlapReranker`, `ExtractiveLLM`, `LexicalJudge`,
`MemoryCache`, in-memory Qdrant. They implement the real protocols, so the default
test suite (≈10 s) exercises real behaviour end to end with no network, and the whole
stack can be demoed without an API key.

## D-024 · Packaging

Multi-stage `uv` build, non-root user, healthcheck, models on a named volume
(downloaded once). Serving image ≈ 760 MB (no torch); `--build-arg EXTRAS=ingest`
adds Docling. Compose: qdrant (pinned v1.19.1, matches client), redis (bounded LRU),
api, and an `ingest` job profile.

## D-025 · Logging streams

API logs JSON to stdout (container convention); the CLI logs to stderr so stdout stays
pipeable. The stream is resolved at call time (a cached `sys.stdout` broke when the
test runner swapped streams).

## D-026 · Real corpus: polite, resumable SEBI downloader

`ragsys fetch-sebi --limit 500` pages through sebi.gov.in → Legal → Circulars (robots.txt
permits it). One request at a time, 1.5 s apart, exponential back-off, PDF magic-byte
check, and a `manifest.jsonl` that makes re-runs resume (skip done, retry failed) and
records HTML-only circulars as `no_pdf`. The listing's AJAX endpoint rejects requests
without the session cookie/referer of the listing page, so page 0 is fetched first.
Synthetic samples moved to `data/samples/` so they never mix with the real corpus.

## D-027 · Titles and provenance from sidecar metadata

**Found on real data:** every SEBI circular's first heading is literally "CIRCULAR", so
"first heading = title" made every citation read "CIRCULAR". **Decision:** (1) a
`manifest.jsonl` next to the PDFs (written by the downloader) supplies the official
title, issue date and URL, applied after parsing so the parse cache stays parser-pure;
(2) without a sidecar, the `Sub:` line is the title and generic headings are skipped.
Chunks and citations carry `doc_date` and `source_url`; passage headers show the date,
which matters because circulars supersede each other.

## D-028 · Credentials

Blank values in `.env` (`ANTHROPIC_API_KEY=`) are treated as unset, so the Anthropic
SDK's normal credential discovery still applies. The app never reads other tools'
credential stores: Claude Code's login is scoped to Claude Code, and a project like this
should use its own Console API key (separately revocable, with its own usage/cost).

## D-029 · Adaptive OCR

Docling's default pipeline OCRs every page: **1.74 s/page** on real circulars vs
**0.28 s/page** without OCR (24-page sample) — 6× slower, and pointless for
born-digital PDFs. But the corpus does contain scans (e.g. the 31 Jul 2026 circular on
digital-accessibility timelines has no text layer; PyMuPDF extracts 0 characters).
**Decision:** a cheap PyMuPDF check (< 100 chars/page ⇒ scanned) routes only scanned
PDFs through the OCR pipeline. The scanned circular now yields its full text and title.

## D-030 · Superscript ordinals, and why the parse cache needs a version

**Found on real data.** Building the seed test set, one item was rejected by the
verbatim-quote check (D-016): the 28 Aug 2026 ETF circular reads *"shall now come into
effect from 7 th September 2026"*. The ordinal suffix is a separate text run in the PDF
(it is superscripted), so **every** parser sees the gap. Effective dates are the most
cited fact in a circular, and the gap both emits a useless bare `th` BM25 token and
leaks into quoted answers.
**Decision.** `normalize_text` re-attaches `(?<=\d)\s+(st|nd|rd|th)\b`. The lookbehind
and the trailing `\b` keep words that merely start like an ordinal intact — `Regulation
23 the`, `Chapter 5 Standard`, `10 thousand` — which is what the tests pin.

**The second-order bug.** The parse cache was keyed by PDF content alone, so changing
the normaliser would have silently kept serving documents produced by the old one — and
stale canonical text propagates into chunks, embeddings, citations *and* gold evidence
spans with nothing to signal it. **Decision:** the cache key carries
`PARSE_SCHEMA_VERSION`, bumped whenever a parser, the element builder, normalisation or
the `Document` schema changes; a test asserts a bump forces a re-parse. This is the
ingestion-side analogue of the index version (D-010) and the cache fingerprint (D-018):
**every derived artefact is keyed by everything that produced it.**

## D-031 · First full evaluation on the real corpus (496 circulars, 28-item seed set)

Run `20261009T063824Z` (git `ee34676`): 3 chunkers × 5 retrieval variants, extractive
answerer, lexical judge. Indexes: fixed 6 466 / recursive 6 808 / structural 7 986
chunks. Headline numbers (best per column in the report):

| finding | evidence |
|---|---|
| **BM25 alone beats dense alone** on regulatory text | R@1 0.729 vs 0.500 (structural), R@5 0.958 vs 0.854 — circulars are identifier- and date-heavy, exactly where embeddings are weak (D-006 vindicated) |
| **Fusion + rerank is what reaches R@5 = 1.000** | hybrid_rerank hits 1.000 R@5 on fixed and structural; no single retriever does |
| **bge-reranker wins precision, loses latency** | best R@1 0.833 / MRR 0.917 (fixed) but p50 3.5 s vs 1.2 s for MiniLM-L12 — keeps MiniLM the serving default (D-009), bge an offline/quality option |
| **Abstention gate: no false refusals, weak recall** | false_abstain_rate 0.000 everywhere; abstain_recall only 0.25–0.50 — the synthetic-corpus thresholds (D-012) are too conservative on real data, and the extractive answerer cannot perform layer-2 instructed abstention |

`correctness` ≈ 0.30 is the lexical token-F1 proxy against reference answers — it is a
floor, not a quality claim (D-014); re-judge with RAGAS + a real key before quoting.
Test-set hash `35847bb67685`; items are hand-written seeds pending human review (D-016).

---

## Known limitations / next steps

* RAGAS judge and test-set generation were verified up to the API boundary (request
  shape, mocked transport); run them with a real `ANTHROPIC_API_KEY` before quoting
  answer-quality numbers.
* Multi-column layouts: the PyMuPDF fallback reads top-to-bottom.
* No auth / rate limiting on the API (put it behind a gateway).
* Orphaned collections from crashed index builds are not garbage-collected.
* Thresholds (D-012) must be re-calibrated on the real corpus: D-031 measured
  abstain_recall 0.25–0.50 at zero false refusals — raise `min_rerank_score` /
  `min_dense_score` until abstain_recall rises without false abstentions appearing.
