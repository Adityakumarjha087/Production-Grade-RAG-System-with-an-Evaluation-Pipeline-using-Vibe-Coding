"""Cache, cost, tracing, config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fakeredis
import pytest

from ragsys.cache import MemoryCache, NullCache, RedisCache, cache_key, normalize_query
from ragsys.config import ModelPrice, Settings, deep_merge, load_config
from ragsys.factory import build_cache
from ragsys.models import Answer, Usage
from ragsys.observability.cost import compute_cost, find_price
from ragsys.observability.tracing import LangfuseSpan, NoopSpan, NoopTracer, make_tracer
from tests.conftest import ROOT, offline_config


class TestCacheKeys:
    def test_normalisation(self) -> None:
        assert normalize_query("  What is  T+1 ?? ") == normalize_query("what is t+1")

    def test_key_depends_on_config_and_index(self) -> None:
        base = cache_key("q", "cfg1", "v1")
        assert base == cache_key("Q?", "cfg1", "v1")
        assert base != cache_key("q", "cfg2", "v1")
        assert base != cache_key("q", "cfg1", "v2")
        assert base.startswith("rag:answer:")


ANSWER = Answer(query="q", text="a [1]", usage=Usage(input_tokens=3))


class TestMemoryCache:
    async def test_roundtrip_and_lru(self) -> None:
        cache = MemoryCache(max_entries=2)
        for k in ("a", "b", "c"):
            await cache.set(k, ANSWER, 60)
        assert await cache.get("a") is None  # evicted
        assert await cache.get("c") == ANSWER

    async def test_ttl_expiry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cache = MemoryCache()
        await cache.set("k", ANSWER, 10)
        import ragsys.cache as mod

        now = mod.time.monotonic()
        monkeypatch.setattr(mod.time, "monotonic", lambda: now + 11)
        assert await cache.get("k") is None


class TestRedisCache:
    def _cache(self) -> RedisCache:
        cache = RedisCache("redis://unused:6379/0")
        cache._redis = fakeredis.FakeAsyncRedis()  # type: ignore[assignment]
        return cache

    async def test_roundtrip_with_ttl(self) -> None:
        cache = self._cache()
        await cache.set("k", ANSWER, 30)
        assert await cache.get("k") == ANSWER
        assert 0 < await cache._redis.ttl("k") <= 30
        assert await cache.ping()

    async def test_failures_degrade_to_miss(self) -> None:
        cache = RedisCache("redis://127.0.0.1:1/0")  # nothing listening
        await cache.set("k", ANSWER, 30)
        assert await cache.get("k") is None
        assert not await cache.ping()
        await cache.close()


async def test_null_cache() -> None:
    cache = NullCache()
    await cache.set("k", ANSWER, 1)
    assert await cache.get("k") is None


def test_build_cache_requires_redis_url() -> None:
    with pytest.raises(ValueError, match="REDIS_URL"):
        build_cache(offline_config(cache={"backend": "redis"}), Settings(_env_file=None))  # type: ignore[call-arg]


class TestCost:
    PRICING = {
        "claude-opus-5-5": ModelPrice(input=4, output=20, cache_read=0.2, cache_write=5),
        "claude-opus": ModelPrice(input=99, output=99),
    }

    def test_compute(self) -> None:
        usage = Usage(input_tokens=1_000_000, output_tokens=100_000, cache_read_tokens=1_000_000)
        assert compute_cost(usage, "claude-opus-5-5", self.PRICING) == pytest.approx(4 + 2 + 0.2)

    def test_longest_prefix_match_and_unknown(self) -> None:
        assert find_price("claude-opus-5-5", self.PRICING) is self.PRICING["claude-opus-5-5"]
        assert find_price("claude-opus-9", self.PRICING) is self.PRICING["claude-opus"]
        assert compute_cost(Usage(input_tokens=10), "gpt-x", self.PRICING) == 0.0

    def test_base_config_prices_default_models(self) -> None:
        cfg = load_config(ROOT / "configs/base.yaml")
        for model in (cfg.generation.model, cfg.eval.judge_model, cfg.eval.testgen_model):
            assert find_price(model, cfg.pricing) is not None


class TestTracing:
    def test_noop_without_keys(self) -> None:
        assert isinstance(make_tracer(True, None, None, "h"), NoopTracer)
        assert isinstance(make_tracer(False, "pk", "sk", "h"), NoopTracer)

    def test_noop_span_api(self) -> None:
        with NoopTracer().start_trace("t") as span:
            with span.child("c", as_type="retriever") as child:
                child.update(output=1)
        assert span.trace_id is None

    def test_langfuse_span_adapter_forwards_and_swallows_errors(self) -> None:
        calls: list[tuple[str, Any]] = []

        class FakeObs:
            trace_id = "t1"

            def start_observation(self, **kw: Any) -> FakeObs:
                calls.append(("start", kw))
                return FakeObs()

            def update(self, **kw: Any) -> None:
                raise RuntimeError("network down")

            def end(self) -> None:
                calls.append(("end", None))

        span = LangfuseSpan(FakeObs())
        assert span.trace_id == "t1"
        with span.child("gen", as_type="generation", model="m", input="x") as child:
            child.update(output="y")  # raises inside -> swallowed
        assert calls[0] == (
            "start",
            {"name": "gen", "as_type": "generation", "input": "x", "model": "m"},
        )
        assert ("end", None) in calls

    def test_span_records_errors(self) -> None:
        updates: list[dict[str, Any]] = []

        class Obs:
            trace_id = None

            def update(self, **kw: Any) -> None:
                updates.append(kw)

            def end(self) -> None:
                pass

        with pytest.raises(ValueError), LangfuseSpan(Obs()):
            raise ValueError("boom")
        assert updates == [{"level": "ERROR", "status_message": "ValueError: boom"}]

    def test_child_of_failed_start_is_noop(self) -> None:
        class Broken:
            trace_id = None

            def start_observation(self, **kw: Any) -> None:
                raise RuntimeError

        assert isinstance(LangfuseSpan(Broken()).child("x"), NoopSpan)


class TestConfig:
    def test_base_loads(self) -> None:
        cfg = load_config(ROOT / "configs/base.yaml")
        assert cfg.chunking.strategy == "structural"
        assert cfg.collection_name() == "sebi_structural_bge-base-en-v1-5"

    @pytest.mark.parametrize("name", ["local.yaml", "docker.yaml"])
    def test_extends(self, name: str) -> None:
        cfg = load_config(ROOT / "configs" / name)
        assert cfg.generation.provider == "extractive"
        assert cfg.pricing  # inherited from base

    def test_circular_extends(self, tmp_path: Path) -> None:
        (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
        (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
        with pytest.raises(ValueError, match="Circular"):
            load_config(tmp_path / "a.yaml")

    def test_blank_secrets_are_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "  ")
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.anthropic_api_key is None and settings.langfuse_secret_key is None

    def test_deep_merge(self) -> None:
        assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}}) == {"a": {"b": 1, "c": 3}}

    def test_fingerprint_tracks_behaviour_only(self) -> None:
        a = offline_config()
        assert (
            a.fingerprint()
            == offline_config(paths={"raw_dir": "elsewhere"}, name="x").fingerprint()
        )
        assert a.fingerprint() != offline_config(retrieval={"final_k": 3}).fingerprint()

    def test_matrix_file_expands(self) -> None:
        from ragsys.evaluation.runner import load_matrix

        variants = load_matrix(ROOT / "configs/experiments/matrix.yaml")
        assert len(variants) == 15
        assert {v.chunker for v in variants} == {"fixed", "recursive", "structural"}
