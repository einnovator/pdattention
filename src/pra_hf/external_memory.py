"""Secure read-side lifecycle for external PRA memory.

The module deliberately stops at read-only admission.  Cheap descriptors name
cold resources, native encodings form warm memory, and selected native K/V form
hot memory.  Consolidation, model-authored writes, and forgetting policies are
outside this boundary.
"""

from __future__ import annotations

import hashlib
import inspect
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Protocol, Sequence
from urllib.parse import unquote, urlparse


EncodingMode = Literal["eager", "lazy", "cached_lazy", "metadata_only"]
CacheScope = Literal["session", "user", "tenant", "global_public"]
MemoryTier = Literal["cold", "warm", "hot"]


@dataclass(frozen=True, repr=False)
class AuthContext:
    """Opaque authorization state passed only to resource resolvers.

    ``credential_provider`` is a runtime handle, never serialized, logged, or
    exposed to a model.  Resolvers may call it immediately before source access.
    """

    tenant_id: str
    user_id: str
    session_id: str
    authorization_scopes: frozenset[str] = frozenset()
    credential_provider: Callable[[str, str], Any] | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    def __repr__(self) -> str:
        return (
            "AuthContext(tenant_id={!r}, user_id={!r}, session_id={!r}, "
            "authorization_scopes={!r}, credential_provider=<opaque>)"
        ).format(
            self.tenant_id,
            self.user_id,
            self.session_id,
            self.authorization_scopes,
        )

    def credentials_for(self, resolver: str, uri: str) -> Any:
        """Resolve credentials at the source boundary without retaining them."""
        if self.credential_provider is None:
            return None
        return self.credential_provider(resolver, uri)


@dataclass(frozen=True)
class ResourceStat:
    """Versioned metadata returned after a resolver authorizes a resource."""

    uri: str
    resolver: str
    version: str
    size_bytes: int
    mime_type: str = "application/octet-stream"
    title: str | None = None
    content_hash: str | None = None
    etag: str | None = None
    is_public: bool = False


@dataclass(frozen=True)
class EncodingContext:
    """Fingerprint inputs that determine whether native encoding is reusable."""

    model_fingerprint: str
    tokenizer_fingerprint: str
    encoding_config: dict[str, Any]

    @property
    def config_fingerprint(self) -> str:
        payload = repr(sorted(self.encoding_config.items())).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:24]


@dataclass(frozen=True)
class NativeEncoding:
    """Warm, model-specific memory; opaque payload commonly contains native K/V."""

    uri: str
    source_version: str
    model_fingerprint: str
    tokenizer_fingerprint: str
    config_fingerprint: str
    token_count: int
    byte_count: int
    pra_gists: tuple[Any, ...] = ()
    logical_offsets: tuple[tuple[int, int], ...] = ()
    payload: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class HotMemoryHandle:
    """Accelerator-resident selection derived from one warm encoding."""

    uri: str
    source_version: str
    selected_token_count: int
    byte_count: int
    payload: Any = field(default=None, repr=False, compare=False)


@dataclass
class ResourceRecord:
    """Session-visible resource registration without credentials or source bytes."""

    uri: str
    resolver: str
    encoding_mode: EncodingMode
    cache_scope: CacheScope
    external_gist: str | None = None
    metadata: ResourceStat | None = None
    tier: MemoryTier = "cold"


@dataclass
class PRASession:
    """Isolated read-side state shared by routing, admission, and generation."""

    session_id: str
    user_id: str
    tenant_id: str
    auth_context: AuthContext
    memory_policy: dict[str, Any] = field(default_factory=dict)
    cache_scope: CacheScope = "session"
    adaptive_controller_state: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, ResourceRecord] = field(default_factory=dict)
    admitted_resources: set[str] = field(default_factory=set)
    warm_handles: dict[str, NativeEncoding] = field(default_factory=dict, repr=False)
    hot_handles: dict[str, HotMemoryHandle] = field(default_factory=dict, repr=False)
    closed: bool = False

    def __post_init__(self) -> None:
        if self.auth_context.session_id != self.session_id:
            raise ValueError("AuthContext and PRASession session IDs must match.")
        if self.auth_context.user_id != self.user_id:
            raise ValueError("AuthContext and PRASession user IDs must match.")
        if self.auth_context.tenant_id != self.tenant_id:
            raise ValueError("AuthContext and PRASession tenant IDs must match.")

    def safe_snapshot(self) -> dict[str, Any]:
        """Return artifact-safe state with no auth object, credentials, or payloads."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "cache_scope": self.cache_scope,
            "resource_uris": sorted(self.resources),
            "admitted_resource_uris": sorted(self.admitted_resources),
            "warm_resource_uris": sorted(self.warm_handles),
            "hot_resource_uris": sorted(self.hot_handles),
            "closed": self.closed,
        }


class ResourceResolver(Protocol):
    """Plugin boundary for authenticated metadata, fetch, and cheap descriptors."""

    async def stat(
        self,
        uri: str,
        auth_context: AuthContext,
        session: PRASession,
    ) -> ResourceStat: ...

    async def fetch(
        self,
        uri: str,
        auth_context: AuthContext,
        session: PRASession,
        byte_range: tuple[int, int] | None = None,
    ) -> bytes: ...

    async def external_gist(
        self,
        uri: str,
        metadata: ResourceStat,
        auth_context: AuthContext,
        session: PRASession,
    ) -> str: ...


class FileResourceResolver:
    """Local ``file:`` resolver; deployment code should apply an allow-list."""

    name = "file"

    @staticmethod
    def _path(uri: str) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme not in {"", "file"}:
            raise ValueError(f"File resolver cannot open {parsed.scheme!r} URI.")
        raw = unquote(parsed.path)
        if parsed.netloc:
            raw = f"//{parsed.netloc}{raw}"
        if len(raw) >= 3 and raw[0] == "/" and raw[2] == ":":
            raw = raw[1:]
        return Path(raw if parsed.scheme else uri).resolve()

    async def stat(self, uri, auth_context, session) -> ResourceStat:
        _ = auth_context.credentials_for(self.name, uri), session
        path = self._path(uri)
        status = path.stat()
        version = f"{status.st_mtime_ns}:{status.st_size}"
        return ResourceStat(
            uri=uri,
            resolver=self.name,
            version=version,
            size_bytes=status.st_size,
            title=path.name,
        )

    async def fetch(self, uri, auth_context, session, byte_range=None) -> bytes:
        _ = auth_context.credentials_for(self.name, uri), session
        data = self._path(uri).read_bytes()
        if byte_range is None:
            return data
        start, stop = byte_range
        return data[start:stop]

    async def external_gist(self, uri, metadata, auth_context, session) -> str:
        _ = auth_context, session
        return metadata.title or Path(urlparse(uri).path).name


class ResolverRegistry:
    """Map URI schemes or explicit resolver names to resolver plugins."""

    def __init__(self) -> None:
        self._resolvers: dict[str, ResourceResolver] = {}

    def register(self, name: str, resolver: ResourceResolver) -> None:
        if not name:
            raise ValueError("Resolver name cannot be empty.")
        self._resolvers[name.lower()] = resolver

    def resolve(self, uri: str, explicit: str | None = None) -> tuple[str, ResourceResolver]:
        name = (explicit or urlparse(uri).scheme or "file").lower()
        try:
            return name, self._resolvers[name]
        except KeyError as error:
            raise KeyError(f"No resource resolver registered for {name!r}.") from error


@dataclass(frozen=True)
class NativeCacheKey:
    security_scope: str
    uri: str
    source_version: str
    model_fingerprint: str
    tokenizer_fingerprint: str
    config_fingerprint: str


@dataclass
class LifecycleMetrics:
    """Read-side counters with cache tiers and latency components kept separate."""

    candidate_resources: int = 0
    admitted_resources: int = 0
    descriptor_hits: int = 0
    descriptor_misses: int = 0
    source_hits: int = 0
    source_misses: int = 0
    native_hits: int = 0
    native_misses: int = 0
    hot_hits: int = 0
    hot_misses: int = 0
    fetched_bytes: int = 0
    newly_encoded_tokens: int = 0
    external_retrieval_seconds: float = 0.0
    stat_seconds: float = 0.0
    fetch_seconds: float = 0.0
    native_encode_seconds: float = 0.0
    materialization_seconds: float = 0.0

    def snapshot(self) -> dict[str, int | float]:
        """Return flat artifact-safe counters and hit rates."""
        values = asdict(self)
        for name in ("descriptor", "source", "native", "hot"):
            hits = int(values[f"{name}_hits"])
            misses = int(values[f"{name}_misses"])
            values[f"{name}_hit_rate"] = hits / max(hits + misses, 1)
        return values


Encoder = Callable[
    [bytes, ResourceStat, EncodingContext],
    NativeEncoding | Awaitable[NativeEncoding],
]
Materializer = Callable[
    [NativeEncoding, Sequence[int] | None],
    HotMemoryHandle | Awaitable[HotMemoryHandle],
]


async def _await_if_needed(value):
    return await value if inspect.isawaitable(value) else value


class ExternalMemoryManager:
    """Coordinate secure cold-to-warm-to-hot admission for PRA sessions."""

    def __init__(
        self,
        *,
        encoding_context: EncodingContext,
        encoder: Encoder,
        materializer: Materializer,
        resolvers: ResolverRegistry | None = None,
    ) -> None:
        self.encoding_context = encoding_context
        self.encoder = encoder
        self.materializer = materializer
        self.resolvers = resolvers or ResolverRegistry()
        self.descriptor_cache: dict[tuple[str, str], ResourceStat] = {}
        self.source_cache: dict[tuple[str, str, str], bytes] = {}
        self.native_cache: dict[NativeCacheKey, NativeEncoding] = {}
        self.hot_cache: dict[NativeCacheKey, HotMemoryHandle] = {}
        self.metrics = LifecycleMetrics()

    @staticmethod
    def _require_open(session: PRASession) -> None:
        if session.closed:
            raise RuntimeError("PRA session is closed.")

    @staticmethod
    def _scope_identity(
        session: PRASession,
        scope: CacheScope,
        metadata: ResourceStat,
    ) -> str:
        if scope == "session":
            return f"tenant:{session.tenant_id}/user:{session.user_id}/session:{session.session_id}"
        if scope == "user":
            return f"tenant:{session.tenant_id}/user:{session.user_id}"
        if scope == "tenant":
            return f"tenant:{session.tenant_id}"
        if scope == "global_public":
            if not metadata.is_public:
                raise PermissionError("Private resources cannot use the global-public cache.")
            return "global-public"
        raise ValueError(f"Unsupported cache scope: {scope}")

    def _native_key(
        self,
        session: PRASession,
        record: ResourceRecord,
        metadata: ResourceStat,
    ) -> NativeCacheKey:
        return NativeCacheKey(
            self._scope_identity(session, record.cache_scope, metadata),
            record.uri,
            metadata.version,
            self.encoding_context.model_fingerprint,
            self.encoding_context.tokenizer_fingerprint,
            self.encoding_context.config_fingerprint,
        )

    async def _stat(
        self,
        session: PRASession,
        record: ResourceRecord,
    ) -> tuple[ResourceStat, bool]:
        """Reauthorize on every access, then classify descriptor reuse."""
        self._require_open(session)
        name, resolver = self.resolvers.resolve(record.uri, record.resolver)
        started = time.perf_counter()
        metadata = await resolver.stat(record.uri, session.auth_context, session)
        self.metrics.stat_seconds += time.perf_counter() - started
        if metadata.resolver != name:
            raise ValueError("Resolver metadata does not match the registered resolver.")
        scope = self._scope_identity(session, record.cache_scope, metadata)
        descriptor_key = (scope, record.uri)
        previous = self.descriptor_cache.get(descriptor_key)
        hit = previous is not None and previous.version == metadata.version
        if hit:
            self.metrics.descriptor_hits += 1
        else:
            self.metrics.descriptor_misses += 1
            self.descriptor_cache[descriptor_key] = metadata
            self._purge_stale(scope, record.uri, metadata.version)
        record.metadata = metadata
        if not hit:
            record.tier = "cold"
            session.warm_handles.pop(record.uri, None)
            session.hot_handles.pop(record.uri, None)
        return metadata, hit

    def _purge_stale(self, scope: str, uri: str, current_version: str) -> None:
        for key in list(self.source_cache):
            if key[0] == scope and key[1] == uri and key[2] != current_version:
                del self.source_cache[key]
        for cache in (self.native_cache, self.hot_cache):
            for key in list(cache):
                if (
                    key.security_scope == scope
                    and key.uri == uri
                    and key.source_version != current_version
                ):
                    del cache[key]

    async def add_reference(
        self,
        session: PRASession,
        *,
        uri: str,
        encoding_mode: EncodingMode = "lazy",
        external_gist: str | None = None,
        resolver: str | None = None,
        cache_scope: CacheScope | None = None,
    ) -> ResourceRecord:
        """Register a cold reference and honor eager/cached-lazy semantics."""
        if encoding_mode not in {"eager", "lazy", "cached_lazy", "metadata_only"}:
            raise ValueError(f"Unsupported encoding mode: {encoding_mode}")
        resolver_name, resolver_plugin = self.resolvers.resolve(uri, resolver)
        _ = resolver_plugin
        record = ResourceRecord(
            uri=uri,
            resolver=resolver_name,
            encoding_mode=encoding_mode,
            cache_scope=cache_scope or session.cache_scope,
            external_gist=external_gist,
        )
        session.resources[uri] = record
        metadata, _ = await self._stat(session, record)
        if record.external_gist is None:
            record.external_gist = await resolver_plugin.external_gist(
                uri,
                metadata,
                session.auth_context,
                session,
            )
        if encoding_mode == "eager":
            await self.ensure_warm(session, uri)
        elif encoding_mode == "cached_lazy":
            key = self._native_key(session, record, metadata)
            if key in self.native_cache:
                self.metrics.native_hits += 1
                session.warm_handles[uri] = self.native_cache[key]
                record.tier = "warm"
        return record

    async def route_candidates(
        self,
        session: PRASession,
        query: str,
        *,
        max_candidates: int = 8,
    ) -> list[tuple[str, float]]:
        """Rank cold descriptors with a deterministic lexical reference router."""
        self._require_open(session)
        started = time.perf_counter()
        query_terms = set(query.casefold().split())
        rows = []
        for record in session.resources.values():
            metadata, _ = await self._stat(session, record)
            text = " ".join(
                value for value in (record.external_gist, metadata.title, record.uri) if value
            )
            terms = set(text.casefold().split())
            score = len(query_terms & terms) / max(len(query_terms), 1)
            rows.append((record.uri, float(score)))
        rows.sort(key=lambda row: (-row[1], row[0]))
        selected = rows[:max_candidates]
        self.metrics.candidate_resources += len(selected)
        self.metrics.external_retrieval_seconds += time.perf_counter() - started
        return selected

    async def admit(
        self,
        session: PRASession,
        candidates: Sequence[tuple[str, float]],
        *,
        max_admitted: int = 2,
        threshold: float = 0.0,
    ) -> list[NativeEncoding]:
        """Promote bounded candidates to warm native memory."""
        admitted = []
        for uri, score in candidates:
            if len(admitted) >= max_admitted or score < threshold:
                continue
            record = session.resources[uri]
            if record.encoding_mode == "metadata_only":
                continue
            admitted.append(await self.ensure_warm(session, uri))
            session.admitted_resources.add(uri)
        self.metrics.admitted_resources += len(admitted)
        return admitted

    async def ensure_warm(self, session: PRASession, uri: str) -> NativeEncoding:
        """Fetch and encode a resource only when a valid scoped warm entry is absent."""
        self._require_open(session)
        record = session.resources[uri]
        if record.encoding_mode == "metadata_only":
            raise PermissionError("metadata_only resources cannot be native-encoded.")
        metadata, _ = await self._stat(session, record)
        key = self._native_key(session, record, metadata)
        cached = self.native_cache.get(key)
        if cached is not None:
            self.metrics.native_hits += 1
            session.warm_handles[uri] = cached
            record.tier = "warm"
            return cached
        self.metrics.native_misses += 1

        source_key = (key.security_scope, uri, metadata.version)
        source = self.source_cache.get(source_key)
        if source is None:
            self.metrics.source_misses += 1
            _, resolver = self.resolvers.resolve(uri, record.resolver)
            started = time.perf_counter()
            source = await resolver.fetch(uri, session.auth_context, session)
            self.metrics.fetch_seconds += time.perf_counter() - started
            self.metrics.fetched_bytes += len(source)
            self.source_cache[source_key] = source
        else:
            self.metrics.source_hits += 1

        started = time.perf_counter()
        encoding = await _await_if_needed(self.encoder(source, metadata, self.encoding_context))
        self.metrics.native_encode_seconds += time.perf_counter() - started
        if encoding.source_version != metadata.version or encoding.uri != uri:
            raise ValueError("Encoder returned an encoding for a different source identity.")
        self.metrics.newly_encoded_tokens += int(encoding.token_count)
        self.native_cache[key] = encoding
        session.warm_handles[uri] = encoding
        record.tier = "warm"
        return encoding

    async def ensure_hot(
        self,
        session: PRASession,
        uri: str,
        *,
        selected_token_ids: Sequence[int] | None = None,
    ) -> HotMemoryHandle:
        """Materialize selected native state after revalidating warm identity."""
        warm = await self.ensure_warm(session, uri)
        record = session.resources[uri]
        metadata = record.metadata
        assert metadata is not None
        key = self._native_key(session, record, metadata)
        if selected_token_ids is None and key in self.hot_cache:
            self.metrics.hot_hits += 1
            handle = self.hot_cache[key]
        else:
            self.metrics.hot_misses += 1
            started = time.perf_counter()
            handle = await _await_if_needed(self.materializer(warm, selected_token_ids))
            self.metrics.materialization_seconds += time.perf_counter() - started
            if selected_token_ids is None:
                self.hot_cache[key] = handle
        session.hot_handles[uri] = handle
        record.tier = "hot"
        return handle

    def teardown_session(self, session: PRASession) -> None:
        """Remove ephemeral cache/state while preserving broader authorized caches."""
        prefix = f"tenant:{session.tenant_id}/user:{session.user_id}/session:{session.session_id}"
        for key in list(self.descriptor_cache):
            if key[0] == prefix:
                del self.descriptor_cache[key]
        for key in list(self.source_cache):
            if key[0] == prefix:
                del self.source_cache[key]
        for cache in (self.native_cache, self.hot_cache):
            for key in list(cache):
                if key.security_scope == prefix:
                    del cache[key]
        session.resources.clear()
        session.admitted_resources.clear()
        session.warm_handles.clear()
        session.hot_handles.clear()
        session.adaptive_controller_state.clear()
        session.closed = True
