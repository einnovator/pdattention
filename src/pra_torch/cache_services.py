"""Configurable resolver/cache services for PRA reference memory."""

from __future__ import annotations

from collections.abc import Iterable

from data.tokenizer import PRATokenizer
from .config import CacheServiceConfig, ResolverServiceConfig
from .memory import PRAMemoryCache, PRASimpleMemoryCache
from .resolution import RecursiveReferenceCacheBuilder
from .resolver import InMemoryResolver


def create_resolver(config, documents: dict, summaries: dict[str, str] | None = None):
    """Instantiate the configured URI-to-source backend.

    Cache construction depends only on this resolver contract, which keeps
    storage/network concerns outside the model and routing implementation.
    """
    config = ResolverServiceConfig.from_value(config)
    if config.type == "in_memory":
        return InMemoryResolver(documents, summaries, **config.options)
    raise ValueError(f"Unsupported resolver service type: {config.type}")


def create_cache(config) -> PRAMemoryCache:
    """Instantiate the configured encoded-memory storage/routing backend."""
    config = CacheServiceConfig.from_value(config)
    if config.type == "simple":
        return PRASimpleMemoryCache(**config.options)
    raise ValueError(f"Unsupported cache service type: {config.type}")


def collect_reference_metadata(metadata: Iterable[dict]) -> tuple[dict, dict[str, str], list]:
    """Convert collator records into resolver documents, summaries, and roots.

    Dataset references carry text in metadata while the prompt carries only
    lightweight tokens. This function reconstructs the runtime resolver view.
    """
    documents: dict = {}
    summaries: dict[str, str] = {}
    handles = []
    for item in metadata:
        for ref in item["references"]:
            ref_metadata = dict(ref.metadata or {})
            text = str(ref_metadata.get("text", ""))
            documents[ref.uri] = {
                "text": text,
                "summary": ref.summary,
                "reference_table": dict(ref_metadata.get("reference_table") or {}),
                "metadata": {
                    key: value
                    for key, value in ref_metadata.items()
                    if key not in {"text", "reference_table", "documents"}
                },
                "version": ref_metadata.get("version"),
            }
            if ref.summary:
                summaries[ref.uri] = ref.summary
            for child_uri, child_value in dict(ref_metadata.get("documents") or {}).items():
                documents[str(child_uri)] = child_value
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
    attach_to_model: bool = True,
) -> PRAMemoryCache:
    """Build, populate, and attach a PRA memory cache from collator metadata.

    ``resolver_config`` and ``cache_config`` select the service implementations.
    Passing explicit ``resolver`` or ``cache`` instances is useful for tests and
    custom research experiments. ``attach_to_model=False`` restores the model's
    previous cache after construction; recursive encoding may still attach the
    row cache temporarily so a parent can attend to its completed children.
    """
    # Turn batch metadata into services, then resolve each distinct root URI.
    documents, summaries, handles = collect_reference_metadata(metadata)
    resolver = resolver if resolver is not None else create_resolver(resolver_config, documents, summaries)
    cache = cache if cache is not None else create_cache(cache_config)

    previous_cache = model.pra_cache
    builder = RecursiveReferenceCacheBuilder(model, resolver, tokenizer, cache, model.cfg)
    try:
        for handle in dict.fromkeys(ref.uri for ref in handles):
            builder.ensure_cached(handle)
    finally:
        if not attach_to_model:
            model.set_pra_cache(previous_cache)

    # Attach lightweight audit data used by evaluation reports and causal traces.
    cache.resolution_events = list(builder.events)
    cache.dependencies = list(builder.dependencies)
    if attach_to_model:
        model.set_pra_cache(cache)
    return cache
