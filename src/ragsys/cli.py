"""Command line interface: ``ragsys <command>``.

ragsys ingest            parse + chunk + index (one or all strategies)
ragsys query "..."       ask a question from the terminal
ragsys testset generate  LLM-generate pending Q/A items
ragsys testset review    approve / edit / reject items interactively
ragsys eval              run the experiment matrix -> comparison table
ragsys serve             start the API
ragsys sample-data       write the synthetic sample corpus (+ test set)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from ragsys.config import PipelineConfig, Settings, load_config

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
testset_app = typer.Typer(no_args_is_help=True, help="Generate and review the evaluation test set.")
app.add_typer(testset_app, name="testset")

ConfigOpt = Annotated[Path | None, typer.Option("--config", "-c", help="Pipeline YAML.")]


def _setup(config: Path | None) -> tuple[Settings, PipelineConfig]:
    from ragsys.observability.logging import configure_logging

    settings = Settings()
    configure_logging(settings.log_level, json_output=False, stream="stderr")
    return settings, load_config(config or settings.rag_config)


def _documents(cfg: PipelineConfig, parser: str | None = None) -> list:  # type: ignore[type-arg]
    from ragsys.ingestion.parsers import get_parser
    from ragsys.ingestion.pipeline import find_pdfs, parse_corpus

    pdfs = find_pdfs(cfg.paths.raw_dir)
    if not pdfs:
        raise typer.BadParameter(f"no PDFs found under {cfg.paths.raw_dir}")
    name = parser or cfg.parsing.parser
    fallback = (
        get_parser(cfg.parsing.fallback_parser)
        if cfg.parsing.fallback_parser and cfg.parsing.fallback_parser != name
        else None
    )
    report = parse_corpus(pdfs, cfg.paths.parsed_dir, get_parser(name), fallback)
    typer.echo(
        f"parsed {len(report.documents)} docs ({report.from_cache} cached, "
        f"{len(report.used_fallback)} via fallback, {len(report.failed)} failed)"
    )
    return report.documents


@app.command()
def ingest(
    config: ConfigOpt = None,
    strategy: Annotated[
        list[str] | None, typer.Option(help="fixed|recursive|structural|all")
    ] = None,
    parser: Annotated[str | None, typer.Option(help="Override parser (docling|pymupdf).")] = None,
    force: Annotated[bool, typer.Option(help="Rebuild even if the index is up to date.")] = False,
) -> None:
    """Parse PDFs and build the index for one or more chunking strategies."""
    from ragsys.embeddings import get_embedder
    from ragsys.indexing.bm25 import BM25Encoder
    from ragsys.indexing.qdrant_store import make_client
    from ragsys.ingestion.chunkers import STRATEGIES, get_chunker
    from ragsys.ingestion.pipeline import chunk_corpus, index_chunks

    settings, cfg = _setup(config)
    docs = _documents(cfg, parser)
    wanted = list(STRATEGIES) if strategy in (["all"],) else (strategy or [cfg.chunking.strategy])
    client = make_client(settings.qdrant_url, settings.qdrant_api_key)
    embedder = get_embedder(cfg.embedding)
    for name in wanted:
        scfg = cfg.model_copy(
            update={"chunking": cfg.chunking.model_copy(update={"strategy": name})}
        )
        chunks = chunk_corpus(docs, get_chunker(scfg.chunking))
        report = index_chunks(
            chunks,
            alias=scfg.collection_name(),
            client=client,
            embedder=embedder,
            bm25=BM25Encoder(cfg.retrieval.sparse.k1, cfg.retrieval.sparse.b),
            batch_size=cfg.index.upsert_batch_size,
            force=force,
        )
        status = "up to date" if report.skipped else f"built in {report.seconds:.1f}s"
        typer.echo(
            f"[{name}] {report.collection} v{report.version}: {report.num_chunks} chunks, {status}"
        )


@app.command()
def query(
    question: str,
    config: ConfigOpt = None,
    stream: Annotated[bool, typer.Option(help="Print tokens as they arrive.")] = True,
) -> None:
    """Ask a question against the index."""
    from ragsys.factory import build_components
    from ragsys.service import DoneEvent, TokenEvent

    settings, cfg = _setup(config)
    comps = build_components(cfg, settings)

    async def run() -> None:
        async for event in comps.service.stream(question, request_id="cli"):
            if isinstance(event, TokenEvent) and stream:
                typer.echo(event.text, nl=False)
            elif isinstance(event, DoneEvent):
                a = event.answer
                typer.echo(f"\n\n{a.text}\n" if not stream or a.abstained else "\n")
                for c in a.citations:
                    typer.echo(f"  [{c.marker}] {c.doc_title} p.{c.page_start} {c.section}")
                typer.echo(
                    f"\nconfidence={a.confidence} abstained={a.abstained} ({a.abstain_reason}) "
                    f"tokens={a.usage.input_tokens}/{a.usage.output_tokens} cost=${a.cost_usd:.5f} "
                    f"latency={a.latency_ms.get('total')}ms"
                )

    asyncio.run(run())
    comps.tracer.flush()


@testset_app.command("generate")
def generate_testset(
    config: ConfigOpt = None,
    n: Annotated[int, typer.Option(help="Number of items to request.")] = 100,
    out: Annotated[Path | None, typer.Option(help="Output JSONL (default: config path).")] = None,
) -> None:
    """Generate pending Q/A items from the parsed corpus (needs ANTHROPIC_API_KEY)."""
    from ragsys.evaluation.testset import QType, TestsetGenerator, save_testset
    from ragsys.generation.llm.anthropic_llm import AnthropicLLM
    from ragsys.generation.llm.disk_cache import DiskCachedLLM

    settings, cfg = _setup(config)
    docs = _documents(cfg)
    llm = DiskCachedLLM(
        AnthropicLLM(cfg.eval.testgen_model, effort="medium", api_key=settings.anthropic_api_key),
        cfg.paths.llm_cache_dir,
    )
    mix = {QType.FACTUAL: 0.5, QType.TABLE: 0.2, QType.MULTI_HOP: 0.15, QType.UNANSWERABLE: 0.15}
    items = asyncio.run(TestsetGenerator(llm, seed=cfg.eval.seed).generate(docs, n, mix))
    path = out or cfg.paths.testset_path
    save_testset(path, items)
    typer.echo(f"wrote {len(items)} pending items to {path}; next: ragsys testset review")


@testset_app.command("from-specs")
def items_from_specs(
    specs: Annotated[Path, typer.Argument(help="YAML file with hand-written items.")],
    config: ConfigOpt = None,
    out: Annotated[Path | None, typer.Option(help="Output JSONL (default: config path).")] = None,
    approved: Annotated[bool, typer.Option(help="Mark items approved (skip review).")] = False,
) -> None:
    """Build test items from hand-written questions + verbatim quotes."""
    from ragsys.evaluation.testset import ReviewStatus, build_from_specs, load_specs, save_testset

    _, cfg = _setup(config)
    status = ReviewStatus.APPROVED if approved else ReviewStatus.PENDING
    items, errors = build_from_specs(load_specs(specs), _documents(cfg), status=status)
    for err in errors:
        typer.echo(f"  skipped: {err}", err=True)
    path = out or cfg.paths.testset_path
    save_testset(path, items)
    typer.echo(f"wrote {len(items)} {status.value} items to {path} ({len(errors)} skipped)")


@testset_app.command("review")
def review_testset(
    config: ConfigOpt = None,
    path: Annotated[Path | None, typer.Option(help="Test set JSONL.")] = None,
    all_items: Annotated[
        bool, typer.Option("--all", help="Re-review non-pending items too.")
    ] = False,
) -> None:
    """Approve, edit or reject generated items (progress is saved after each one)."""
    from ragsys.evaluation.review import run_review

    _, cfg = _setup(config)
    stats = run_review(
        path or cfg.paths.testset_path,
        ask=lambda m: typer.prompt(m, default="", show_default=False),
        show=typer.echo,
        only_pending=not all_items,
    )
    typer.echo(stats)


@testset_app.command("stats")
def show_testset_stats(
    config: ConfigOpt = None, path: Annotated[Path | None, typer.Option()] = None
) -> None:
    """Counts by review status and question type."""
    from ragsys.evaluation.review import review_stats
    from ragsys.evaluation.testset import load_testset

    _, cfg = _setup(config)
    typer.echo(review_stats(load_testset(path or cfg.paths.testset_path)))


@app.command("eval")
def evaluate(
    config: ConfigOpt = None,
    matrix: Annotated[Path, typer.Option(help="Experiment matrix YAML.")] = Path(
        "configs/experiments/matrix.yaml"
    ),
    answers: Annotated[bool, typer.Option(help="Also generate + judge answers.")] = True,
    subset: Annotated[int | None, typer.Option(help="Answer-eval sample size.")] = None,
    include_pending: Annotated[bool, typer.Option(help="Use unreviewed items too.")] = False,
    only: Annotated[list[str] | None, typer.Option(help="Run only these variant names.")] = None,
) -> None:
    """Run all configurations and write a comparison report."""
    from ragsys.evaluation.answer_metrics import make_judge
    from ragsys.evaluation.report import columns_for, to_markdown, write_report
    from ragsys.evaluation.runner import EvalRunner, load_matrix
    from ragsys.evaluation.testset import ReviewStatus, hash_testset, load_testset
    from ragsys.generation.llm import get_llm
    from ragsys.generation.llm.disk_cache import DiskCachedLLM
    from ragsys.indexing.qdrant_store import make_client

    settings, cfg = _setup(config)
    if subset is not None:
        cfg = cfg.model_copy(update={"eval": cfg.eval.model_copy(update={"answer_subset": subset})})
    items = [
        i
        for i in load_testset(cfg.paths.testset_path)
        if i.status == ReviewStatus.APPROVED
        or (include_pending and i.status == ReviewStatus.PENDING)
    ]
    if not items:
        raise typer.BadParameter("no approved test items (review them or pass --include-pending)")
    variants = [v for v in load_matrix(matrix) if not only or v.name in only]
    llm = (
        DiskCachedLLM(get_llm(cfg.generation, settings.anthropic_api_key), cfg.paths.llm_cache_dir)
        if answers
        else None
    )
    judge = (
        make_judge(cfg.eval.judge_backend, cfg.eval.judge_model, settings.anthropic_api_key)
        if answers
        else None
    )
    runner = EvalRunner(
        cfg,
        settings,
        items,
        _documents(cfg),
        llm=llm,
        judge=judge,
        client=make_client(settings.qdrant_url, settings.qdrant_api_key),
    )
    results = asyncio.run(runner.run(variants, answers=answers))
    meta = {
        "config": cfg.name,
        "matrix": str(matrix),
        "testset": str(cfg.paths.testset_path),
        "testset_hash": hash_testset(items),
        "n_items": len(items),
        "generator": cfg.generation.model
        if cfg.generation.provider == "anthropic"
        else cfg.generation.provider,
        "judge": f"{cfg.eval.judge_backend}:{cfg.eval.judge_model}" if answers else "none",
    }
    out = write_report(results, cfg.paths.results_dir, meta, cfg.model_dump(mode="json"))
    typer.echo(
        "\n" + to_markdown(results, columns_for(results)) + f"\n\nreport: {out / 'report.md'}"
    )


@app.command()
def serve(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
) -> None:
    """Start the API server."""
    import uvicorn

    uvicorn.run(
        "ragsys.api.app:app_factory",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_config=None,
    )


@app.command("fetch-sebi")
def fetch_sebi(
    limit: Annotated[int, typer.Option(help="Number of circular PDFs to download.")] = 500,
    out: Annotated[Path, typer.Option(help="Destination directory.")] = Path("data/raw/sebi"),
    delay: Annotated[float, typer.Option(help="Seconds between requests (be polite).")] = 1.5,
) -> None:
    """Download the newest public SEBI circulars (resumable; see manifest.jsonl)."""
    from ragsys.devtools.sebi_fetch import SebiFetcher

    _setup(None)
    counts = SebiFetcher(out, delay=delay).run(limit)
    typer.echo(f"{counts} -> {out}")


@app.command("sample-data")
def sample_data(
    config: ConfigOpt = None,
    out: Annotated[Path, typer.Option(help="Where to write sample PDFs.")] = Path(
        "data/raw/sample"
    ),
    with_testset: Annotated[
        bool, typer.Option(help="Also write the hand-written test set.")
    ] = True,
) -> None:
    """Write the synthetic sample corpus (and its approved test set)."""
    from ragsys.devtools.sample_corpus import build_sample_corpus, build_sample_testset
    from ragsys.evaluation.testset import save_testset

    _, cfg = _setup(config)
    paths = build_sample_corpus(out)
    typer.echo(f"wrote {len(paths)} sample PDFs to {out}")
    if with_testset:
        docs = [d for d in _documents(cfg) if Path(d.source_path).parent.resolve() == out.resolve()]
        items = build_sample_testset(docs)
        save_testset(cfg.paths.testset_path, items)
        typer.echo(f"wrote {len(items)} approved items to {cfg.paths.testset_path}")


if __name__ == "__main__":
    app()
