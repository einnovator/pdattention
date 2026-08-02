"""Configurable resolver/cache services for PRA reference memory."""

from __future__ import annotations

from collections.abc import Iterable

from data.tokenizer import PRATokenizer
from .config import CacheServiceConfig, ResolverServiceConfig
from .memory import PRAMemoryCache, PRASimpleMemoryCache
from .resolver import InMemoryResolver


def create_resolver(config, documents: dict[str, str], summaries: dict[str, str]):
    """Create a resolver service for the configured backend."""
    config = ResolverServiceConfig.from_value(config)
    if config.type == "in_memory":
        return InMemoryResolver(documents, summaries, **config.options)
    raise ValueError(f"Unsupported resolver service type: {config.type}")


def create_cache(config) -> PRAMemoryCache:
    """Create a PRA memory cache for the configured backend."""
    config = CacheServiceConfig.from_value(config)
    if config.type == "simple":
        return PRASimpleMemoryCache(**config.options)
    raise ValueError(f"Unsupported cache service type: {config.type}")


def collect_reference_metadata(metadata: Iterable[dict]) -> tuple[dict[str, str], dict[str, str], list]:
    """Extract resolver documents, summaries, and handles from batch metadata."""
    documents: dict[str, str] = {}
    summaries: dict[str, str] = {}
    handles = []
    for item in metadata:
        for ref in item["references"]:
            text = str(ref.metadata.get("text", ""))
            documents[ref.uri] = text
            summaries[ref.uri] = ref.summary or ""
            handles.append(ref)
    return documents, summaries, handles


def build_cache_from_metadata(
    model,
    tokenizer: PRATokenizer,
    metadata: list[dict],
    device,
    *,
    resolver_config: ResolverServiceConfig | dict | str | None = None,
    cache_config: CacheServiceConfig | dict | str | None = None,
    resolver=None,
    cache: PRAMemoryCache | None = None,
) -> PRAMemoryCache:
    """Build, populate, and attach a PRA memory cache from collator metadata.

    ``resolver_config`` and ``cache_config`` select the service implementations.
    Passing explicit ``resolver`` or ``cache`` instances is useful for tests and
    custom research experiments.
    """
    documents, summaries, handles = collect_reference_metadata(metadata)
    resolver = resolver if resolver is not None else create_resolver(resolver_config, documents, summaries)
    cache = cache if cache is not None else create_cache(cache_config)

    for handle in handles:
        resolved = resolver.resolve(handle.uri)
        summary = handle.summary or resolved.summary
        entry = model.encode_reference_to_cache(handle.uri, resolved.text, summary, tokenizer, device)
        cache.put(entry)
    model.set_pra_cache(cache)
    return cache
