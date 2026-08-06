"""Bounded child-first construction of recursive PRA reference caches."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass, field

from .memory import CacheBuildState, PRACacheEntry, PRAMemoryCache


class ResolutionBudgetExceeded(RuntimeError):
    """Raised when one recursive root traversal exceeds its shared limits."""

    pass


@dataclass
class ResolutionBudget:
    """Mutable reference/token allowance shared by a recursive cache build."""

    max_total_references: int  # Maximum entries resolved from one root request.
    max_total_tokens: int  # Maximum source tokens encoded across those entries.
    references_used: int = 0  # Entries charged so far.
    tokens_used: int = 0  # Resolved source tokens charged so far.

    def consume(self, token_count: int) -> None:
        """Atomically charge one reference and its source-token count."""
        if self.references_used + 1 > self.max_total_references:
            raise ResolutionBudgetExceeded("max_total_references exhausted")
        if self.tokens_used + token_count > self.max_total_tokens:
            raise ResolutionBudgetExceeded("max_total_tokens exhausted")
        self.references_used += 1
        self.tokens_used += token_count


@dataclass(frozen=True)
class ResolutionEvent:
    """Compact audit event for recursive resolution and cache construction."""

    uri: str  # URI affected by the event.
    event: str  # building, ready, hit, cycle, missing, failed, and so on.
    depth: int  # Number of child edges from the requested root.
    parent_uri: str | None = None  # URI whose local table named this child.
    detail: str | None = None  # Optional failure/budget explanation.


def _fingerprint(value) -> str:
    """Hash JSON-compatible identity material for stale-cache detection."""
    serialized = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


class RecursiveReferenceCacheBuilder:
    """Resolve references child-first and publish only complete cache entries.

    A child is encoded before its parent so recursive mode can let the parent's
    independent encoding attend to already cached child memory. Depth, token,
    reference, cycle, and missing-reference policies bound this expansion.
    """

    def __init__(self, model, resolver, tokenizer, cache: PRAMemoryCache, config):
        """Bind the model-side encoder to resolver/cache services and PRA policy."""
        self.model = model
        self.resolver = resolver  # Converts stable URIs to text and local child tables.
        self.tokenizer = tokenizer  # Defines budget counts and reference tokenization.
        self.cache = cache  # Stores ready layer-specific K/V and lifecycle state.
        self.config = config  # Recursion, chunking, gist, and cache identity settings.
        self.events: list[ResolutionEvent] = []  # Ordered build audit trail.
        self.dependencies: list[dict] = []  # Parent-child edges and their actions.

    def new_budget(self) -> ResolutionBudget:
        """Create one allowance to share across a root and all descendants."""
        return ResolutionBudget(
            max_total_references=self.config.recursive_max_total_references,
            max_total_tokens=self.config.recursive_max_total_tokens,
        )

    def _identity(self, resolved) -> dict:
        """Describe inputs whose change makes an existing cache entry stale."""
        routing_config = {
            key: value
            for key, value in asdict(self.config).items()
            if key
            in {
                "chunking_mode",
                "fixed_chunk_tokens",
                "fixed_chunk_overlap_tokens",
                "gist_mode",
                "max_gists_per_reference",
                "gist_overflow_policy",
                "use_summary",
                "summary_mode",
                "recursive_refs_enabled",
                "recursive_max_depth",
            }
        }
        tokenizer_vocab = getattr(self.tokenizer, "stoi", {})
        return {
            "canonical_uri": resolved.uri,
            "content_fingerprint": _fingerprint({"text": resolved.text, "version": resolved.version}),
            "model_fingerprint": _fingerprint(
                {
                    "class": type(self.model).__name__,
                    "d_model": self.config.d_model,
                    "n_layers": self.config.n_layers,
                    "n_heads": self.config.n_heads,
                }
            ),
            "tokenizer_fingerprint": _fingerprint(tokenizer_vocab),
            "routing_configuration_fingerprint": _fingerprint(routing_config),
        }

    def _missing(self, uri: str, depth: int, error: Exception):
        """Record an unresolved URI and apply skip/warn/error policy."""
        self.events.append(ResolutionEvent(uri, "missing", depth, detail=str(error)))
        policy = self.config.recursive_missing_ref_policy
        if policy == "error":
            raise error
        if policy == "warn":
            warnings.warn(str(error), RuntimeWarning, stacklevel=3)
        return None

    def ensure_cached(
        self,
        uri: str,
        *,
        depth: int = 0,
        ancestry: tuple[str, ...] = (),
        budget: ResolutionBudget | None = None,
        parent_uri: str | None = None,
    ) -> PRACacheEntry | None:
        """Ensure a URI and permitted descendants have ready cache entries.

        The traversal is depth-first. ``ancestry`` detects graph cycles while
        cache ``BUILDING`` state catches re-entrant construction. A matching
        fingerprint is a cache hit; otherwise the stale object is rebuilt.
        """
        budget = budget or self.new_budget()

        # Stop cycles before resolver or model work. ``link_only`` may reuse an
        # already-ready target but never exposes a partially building entry.
        if uri in ancestry:
            self.dependencies.append(
                {"parent_uri": parent_uri, "child_uri": uri, "cyclic": True, "action": self.config.recursive_cycle_policy}
            )
            self.events.append(ResolutionEvent(uri, "cycle", depth, parent_uri=parent_uri))
            if self.config.recursive_cycle_policy == "error":
                raise RuntimeError(f"Recursive reference cycle detected: {' -> '.join((*ancestry, uri))}")
            return self.cache.get(uri) if self.config.recursive_cycle_policy == "link_only" else None
        if hasattr(self.cache, "state") and self.cache.state(uri) == CacheBuildState.BUILDING:
            self.events.append(ResolutionEvent(uri, "reentrant", depth, parent_uri=parent_uri))
            if self.config.recursive_cycle_policy == "error":
                raise RuntimeError(f"Re-entrant reference cache construction for {uri}")
            return None
        # Resolution provides source text, version, summary, and a local token->URI table.
        try:
            resolved = self.resolver.resolve(uri)
        except KeyError as error:
            return self._missing(uri, depth, error)

        # Reuse only when content, model, tokenizer, and routing configuration match.
        identity = self._identity(resolved)
        existing = self.cache.get(uri)
        if existing is not None and all(existing.metadata.get(key) == value for key, value in identity.items()):
            self.events.append(ResolutionEvent(uri, "cache_hit", depth, parent_uri=parent_uri))
            return existing
        if existing is not None and hasattr(self.cache, "invalidate"):
            self.cache.invalidate(uri)

        # Charge the shared traversal budget before making this URI visible as BUILDING.
        token_count = len(self.tokenizer.encode(resolved.text))
        budget.consume(token_count)
        if hasattr(self.cache, "begin_build"):
            self.cache.begin_build(uri)
        self.events.append(ResolutionEvent(uri, "building", depth, parent_uri=parent_uri))
        child_uris = list(dict.fromkeys(resolved.reference_table.values()))
        child_uris = child_uris[: self.config.recursive_max_children_per_reference]
        try:
            # Build children first. Their ready K/V can participate while encoding
            # this parent when recursive_refs_enabled is active.
            if self.config.recursive_refs_enabled and depth < self.config.recursive_max_depth:
                for child_uri in child_uris:
                    self.dependencies.append(
                        {"parent_uri": uri, "child_uri": child_uri, "cyclic": False, "action": "resolve"}
                    )
                    try:
                        self.ensure_cached(
                            child_uri,
                            depth=depth + 1,
                            ancestry=(*ancestry, uri),
                            budget=budget,
                            parent_uri=uri,
                        )
                    except ResolutionBudgetExceeded as error:
                        self.events.append(
                            ResolutionEvent(child_uri, "budget_exhausted", depth + 1, uri, str(error))
                        )
                        break
            elif child_uris:
                reason = "max_depth" if self.config.recursive_refs_enabled else "recursion_disabled"
                for child_uri in child_uris:
                    self.dependencies.append(
                        {"parent_uri": uri, "child_uri": child_uri, "cyclic": False, "action": reason}
                    )

            # Encode all chunks/layers, then atomically publish the complete parent.
            self.model.set_pra_cache(self.cache)
            entry = self.model.encode_reference_to_cache(
                uri,
                resolved.text,
                self.tokenizer,
                next(self.model.parameters()).device,
                metadata={
                    **resolved.metadata,
                    **identity,
                    "version": resolved.version,
                    "summary": resolved.summary,
                    "reference_table": resolved.reference_table,
                    "child_uris": child_uris,
                    "resolution_depth": depth,
                    "resolution_budget_references_used": budget.references_used,
                    "resolution_budget_tokens_used": budget.tokens_used,
                },
                use_pra_memory=self.config.recursive_refs_enabled and any(
                    self.cache.has(child_uri) for child_uri in child_uris
                ),
            )
            self.cache.put(entry)
            self.events.append(ResolutionEvent(uri, "ready", depth, parent_uri=parent_uri))
            return entry
        except Exception as error:
            if hasattr(self.cache, "mark_failed"):
                self.cache.mark_failed(uri, error)
            self.events.append(ResolutionEvent(uri, "failed", depth, parent_uri, str(error)))
            raise
