"""Typed agent resources and auditable discovery policies for Paper 6.5.

This module stops at resource identity selection.  It deliberately contains no
native K/V tensors and never executes a tool: selected identities are handed to
the existing PRA cache/materialization layer after host authorization.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from difflib import SequenceMatcher
from enum import Enum
from typing import Callable, Iterable, Mapping, Sequence


class DiscoveryMode(str, Enum):
    """User-visible starting policies supported by the agent-resource SDK."""

    AUTO = "auto"
    EXPLICIT = "explicit"
    TOKEN = "token"
    INDEX = "index"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


class SideEffectClass(str, Enum):
    """Host-facing execution risk attached to a resource definition."""

    NONE = "none"
    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"


class DiscoveryDecision(str, Enum):
    """Selection outcome; execution authorization is intentionally separate."""

    SELECT = "select"
    ASK = "ask"
    ABSTAIN = "abstain"


_URI = re.compile(
    r"^!!ref:(?P<kind>[a-z][a-z0-9_-]*):(?P<body>.+)!!$",
    flags=re.IGNORECASE,
)
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def normalize_text(text: str) -> str:
    """Return a tokenization-robust comparison form for names and aliases."""

    text = unicodedata.normalize("NFKC", text)
    text = _CAMEL_BOUNDARY.sub(" ", text)
    text = text.replace("_", " ").replace("-", " ")
    return " ".join(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def terms(text: str) -> tuple[str, ...]:
    """Split normalized text into stable word terms."""

    value = normalize_text(text)
    return tuple(value.split()) if value else ()


def _char_ngrams(text: str, width: int = 3) -> frozenset[str]:
    compact = normalize_text(text).replace(" ", "")
    if len(compact) < width:
        return frozenset((compact,)) if compact else frozenset()
    return frozenset(compact[index : index + width] for index in range(len(compact) - width + 1))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / max(left_norm * right_norm, 1e-12)


def hashed_semantic_vector(text: str, dimensions: int = 128) -> tuple[float, ...]:
    """Build a deterministic dependency-free semantic-index control vector.

    The signed feature hash is a lexical semantic control, not a learned model
    embedding.  Production callers can pass a model-backed ``semantic_encoder``.
    """

    vector = [0.0] * dimensions
    for token, frequency in Counter(terms(text)).items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        index = value % dimensions
        sign = 1.0 if (value >> 8) & 1 else -1.0
        vector[index] += sign * (1.0 + math.log(frequency))
    norm = math.sqrt(sum(value * value for value in vector))
    return tuple(value / norm for value in vector) if norm else tuple(vector)


@dataclass(frozen=True)
class DiscoveryHint:
    """Preferred starting policy and whether fallback is prohibited."""

    mode: DiscoveryMode | str = DiscoveryMode.AUTO
    strict: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", DiscoveryMode(self.mode))


@dataclass(frozen=True)
class AgentResource:
    """One typed, versioned tool, skill, artifact, or session object.

    ``uri`` is identity. ``description`` and ``content`` are independently
    versioned source representations that may later be encoded into native K/V.
    Routing hints describe operational properties and are never relevance labels.
    """

    uri: str
    kind: str
    namespace: str
    name: str
    version: str
    description: str
    content: str = ""
    aliases: tuple[str, ...] = ()
    stable_name: bool = True
    indexable: bool = True
    semantic_only: bool = False
    expected_reuse: int = 1
    side_effect_class: SideEffectClass | str = SideEffectClass.NONE
    tenant_id: str = "default"
    revoked: bool = False
    discovery_hint: DiscoveryHint | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        match = _URI.match(self.uri)
        if match is None:
            raise ValueError(f"Invalid typed resource URI: {self.uri}")
        if match.group("kind").casefold() != self.kind.casefold():
            raise ValueError("Resource URI kind does not match resource.kind.")
        if not self.namespace or not self.name or not self.version:
            raise ValueError("namespace, name, and version are required.")
        if self.expected_reuse < 0:
            raise ValueError("expected_reuse cannot be negative.")
        object.__setattr__(self, "aliases", tuple(dict.fromkeys(self.aliases)))
        object.__setattr__(self, "side_effect_class", SideEffectClass(self.side_effect_class))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def search_text(self) -> str:
        """Text visible to discovery indexes, excluding opaque implementation state."""

        semantic_terms = self.metadata.get("semantic_terms", ())
        if isinstance(semantic_terms, str):
            semantic_terms = (semantic_terms,)
        return " ".join(
            (
                self.namespace,
                self.name,
                *self.aliases,
                self.description,
                *(str(value) for value in semantic_terms),
            )
        )

    def fingerprint_payload(self) -> dict[str, object]:
        """Return source fields that invalidate indexes and encoded caches."""

        return {
            "uri": self.uri,
            "version": self.version,
            "description": self.description,
            "content": self.content,
            "aliases": self.aliases,
            "revoked": self.revoked,
            "tenant_id": self.tenant_id,
        }


@dataclass(frozen=True)
class IndexFingerprint:
    """Configuration identity shared by source, index, and later K/V caches."""

    source_digest: str
    tokenizer: str = "normalized-word-v1"
    semantic_encoder: str = "signed-hash-128-v1"
    model: str = "none"
    routing: str = "agent-resource-v1"
    positional: str = "none"

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResourceScore:
    """Channel scores and provenance for one candidate resource."""

    uri: str
    explicit: float = 0.0
    token: float = 0.0
    index: float = 0.0
    semantic: float = 0.0
    hybrid: float = 0.0
    selected_score: float = 0.0
    selected_mode: str = ""
    rank: int | None = None
    provenance: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiscoveryRequest:
    """One resource lookup with request-scoped policy and isolation controls."""

    query: str
    hint: DiscoveryHint | None = None
    namespace: str | None = None
    explicit_reference_uris: tuple[str, ...] = ()
    tenant_id: str = "default"
    top_k: int = 4
    allowed_uris: frozenset[str] | None = None
    side_effecting: bool = False

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")


@dataclass(frozen=True)
class DiscoveryTrace:
    """Complete request/reply record emitted before materialization or execution."""

    request: DiscoveryRequest
    requested_hint: str
    resolved_hint: str
    strict: bool
    executed_path: tuple[str, ...]
    candidates: tuple[ResourceScore, ...]
    selected_uris: tuple[str, ...]
    confidence: float
    margin: float
    decision: DiscoveryDecision
    fallback_count: int
    hint_complied: bool
    index_fingerprint: str
    materialized: bool = False
    execution_authorized: bool = False


@dataclass
class DiscoveryPolicyHints:
    """Hierarchical hints with request > reference > namespace > collection precedence."""

    collection: DiscoveryHint = field(default_factory=DiscoveryHint)
    namespaces: dict[str, DiscoveryHint] = field(default_factory=dict)
    references: dict[str, DiscoveryHint] = field(default_factory=dict)

    def resolve(
        self,
        request: DiscoveryRequest,
        resources: Mapping[str, AgentResource] | None = None,
    ) -> DiscoveryHint:
        if request.hint is not None:
            return request.hint
        for uri in request.explicit_reference_uris:
            if uri in self.references:
                return self.references[uri]
            resource = (resources or {}).get(uri)
            if resource is not None and resource.discovery_hint is not None:
                return resource.discovery_hint
        if request.namespace is not None and request.namespace in self.namespaces:
            return self.namespaces[request.namespace]
        return self.collection


class PersistentResourceIndex:
    """Reusable postings, BM25, n-gram, and semantic sidecar over resource IDs."""

    def __init__(
        self,
        resources: Iterable[AgentResource],
        *,
        semantic_encoder: Callable[[str], Sequence[float]] = hashed_semantic_vector,
        fingerprint_metadata: Mapping[str, str] | None = None,
    ) -> None:
        self.resources = tuple(resources)
        if len({resource.uri for resource in self.resources}) != len(self.resources):
            raise ValueError("Resource URIs must be unique within an index version.")
        self.by_uri = {resource.uri: resource for resource in self.resources}
        self.semantic_encoder = semantic_encoder
        self.exact: dict[str, set[str]] = defaultdict(set)
        self.postings: dict[str, set[str]] = defaultdict(set)
        self.ngram_postings: dict[str, set[str]] = defaultdict(set)
        self.resource_terms: dict[str, tuple[str, ...]] = {}
        self.resource_ngrams: dict[str, frozenset[str]] = {}
        self.semantic_vectors: dict[str, tuple[float, ...]] = {}
        document_frequency: Counter[str] = Counter()
        for resource in self.resources:
            names = (resource.uri, resource.name, *resource.aliases)
            if not resource.semantic_only:
                for name in names:
                    self.exact[normalize_text(name)].add(resource.uri)
            resource_terms = terms(resource.search_text)
            self.resource_terms[resource.uri] = resource_terms
            document_frequency.update(set(resource_terms))
            if resource.indexable and not resource.semantic_only:
                for token in set(resource_terms):
                    self.postings[token].add(resource.uri)
            grams = _char_ngrams(" ".join((resource.name, *resource.aliases)))
            self.resource_ngrams[resource.uri] = grams
            if resource.indexable and not resource.semantic_only:
                for gram in grams:
                    self.ngram_postings[gram].add(resource.uri)
            self.semantic_vectors[resource.uri] = tuple(
                float(value) for value in semantic_encoder(resource.search_text)
            )
        count = max(len(self.resources), 1)
        self.idf = {
            token: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }
        lengths = [len(value) for value in self.resource_terms.values()]
        self.average_length = sum(lengths) / max(len(lengths), 1)
        source = json.dumps(
            [resource.fingerprint_payload() for resource in self.resources],
            sort_keys=True,
            separators=(",", ":"),
        )
        metadata = dict(fingerprint_metadata or {})
        self.fingerprint = IndexFingerprint(
            source_digest=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            tokenizer=metadata.get("tokenizer", "normalized-word-v1"),
            semantic_encoder=metadata.get("semantic_encoder", "signed-hash-128-v1"),
            model=metadata.get("model", "none"),
            routing=metadata.get("routing", "agent-resource-v1"),
            positional=metadata.get("positional", "none"),
        )

    @property
    def estimated_bytes(self) -> int:
        """Return a transparent payload-size estimate, excluding Python overhead."""

        posting_ids = sum(len(values) for values in self.postings.values())
        ngram_ids = sum(len(values) for values in self.ngram_postings.values())
        semantic = sum(len(vector) * 4 for vector in self.semantic_vectors.values())
        text = sum(len(resource.search_text.encode("utf-8")) for resource in self.resources)
        return text + semantic + 8 * (posting_ids + ngram_ids)

    def _eligible(self, request: DiscoveryRequest) -> tuple[AgentResource, ...]:
        values = []
        for resource in self.resources:
            if resource.revoked or resource.tenant_id != request.tenant_id:
                continue
            if request.namespace is not None and resource.namespace != request.namespace:
                continue
            if request.allowed_uris is not None and resource.uri not in request.allowed_uris:
                continue
            values.append(resource)
        return tuple(values)

    def score(self, request: DiscoveryRequest) -> tuple[ResourceScore, ...]:
        """Compute all discovery channels once for policy comparison and fallback."""

        eligible = self._eligible(request)
        eligible_uris = {resource.uri for resource in eligible}
        query_normalized = normalize_text(request.query)
        query_terms = terms(request.query)
        query_grams = _char_ngrams(request.query)
        query_vector = tuple(float(value) for value in self.semantic_encoder(request.query))
        explicit_uris = set(request.explicit_reference_uris)
        explicit_uris.update(re.findall(r"!!ref:[^!]+!!", request.query, flags=re.IGNORECASE))
        exact_hits = self.exact.get(query_normalized, set())
        posting_candidates = set().union(
            *(self.postings.get(token, set()) for token in set(query_terms))
        ) if query_terms else set()
        ngram_candidates = set().union(
            *(self.ngram_postings.get(gram, set()) for gram in query_grams)
        ) if query_grams else set()
        index_candidates = (posting_candidates | ngram_candidates | exact_hits) & eligible_uris
        bm25_raw: dict[str, float] = {}
        for uri in index_candidates:
            document = self.resource_terms[uri]
            frequencies = Counter(document)
            score = 0.0
            for token in set(query_terms):
                frequency = frequencies[token]
                if not frequency:
                    continue
                denominator = frequency + 1.2 * (
                    0.25 + 0.75 * len(document) / max(self.average_length, 1.0)
                )
                score += self.idf.get(token, 0.0) * frequency * 2.2 / denominator
            bm25_raw[uri] = score
        bm25_scale = max(max(bm25_raw.values(), default=0.0), 1e-12)
        rows = []
        for resource in eligible:
            names = (resource.name, *resource.aliases)
            exact = float(
                resource.uri in explicit_uris
                or query_normalized in {normalize_text(name) for name in names}
            )
            name_similarity = max(
                (SequenceMatcher(None, query_normalized, normalize_text(name)).ratio() for name in names),
                default=0.0,
            )
            resource_grams = self.resource_ngrams[resource.uri]
            ngram_overlap = len(query_grams & resource_grams) / max(
                len(query_grams | resource_grams), 1
            )
            query_set = set(query_terms)
            resource_set = set(self.resource_terms[resource.uri])
            term_overlap = len(query_set & resource_set) / max(len(query_set), 1)
            token_score = max(exact, 0.50 * name_similarity + 0.30 * ngram_overlap + 0.20 * term_overlap)
            if resource.semantic_only:
                token_score = 0.0
            index_score = max(exact, bm25_raw.get(resource.uri, 0.0) / bm25_scale)
            if not resource.indexable or resource.semantic_only:
                index_score = 0.0
            semantic = max(0.0, _cosine(query_vector, self.semantic_vectors[resource.uri]))
            hybrid = max(exact, 0.45 * token_score + 0.20 * index_score + 0.35 * semantic)
            rows.append(
                ResourceScore(
                    uri=resource.uri,
                    explicit=float(resource.uri in explicit_uris),
                    token=max(0.0, min(1.0, token_score)),
                    index=max(0.0, min(1.0, index_score)),
                    semantic=max(0.0, min(1.0, semantic)),
                    hybrid=max(0.0, min(1.0, hybrid)),
                    provenance=tuple(
                        channel
                        for channel, value in (
                            ("explicit", resource.uri in explicit_uris),
                            ("exact_name", exact),
                            ("token", token_score > 0.0),
                            ("index", resource.uri in index_candidates),
                            ("semantic", semantic > 0.0),
                        )
                        if value
                    ),
                )
            )
        return tuple(rows)


class ReliabilityCalibrator:
    """Small validation-fitted reliability map for held-out confidence reporting."""

    def __init__(self, bins: Sequence[tuple[float, float]] = ()) -> None:
        self.bins = tuple((float(boundary), float(value)) for boundary, value in bins)

    @classmethod
    def fit(
        cls,
        scores_and_labels: Iterable[tuple[float, bool]],
        *,
        bins: int = 10,
        prior_strength: float = 2.0,
    ) -> "ReliabilityCalibrator":
        rows = sorted((max(0.0, min(1.0, float(score))), bool(label)) for score, label in scores_and_labels)
        if not rows:
            return cls()
        width = max(1, math.ceil(len(rows) / bins))
        values = []
        for start in range(0, len(rows), width):
            chunk = rows[start : start + width]
            mean_score = sum(score for score, _ in chunk) / len(chunk)
            positives = sum(int(label) for _, label in chunk)
            calibrated = (positives + prior_strength * mean_score) / (len(chunk) + prior_strength)
            values.append((chunk[-1][0], calibrated))
        values[-1] = (1.0, values[-1][1])
        return cls(values)

    def __call__(self, score: float) -> float:
        score = max(0.0, min(1.0, float(score)))
        for boundary, value in self.bins:
            if score <= boundary:
                return value
        return score


class ResourceDiscoveryEngine:
    """Execute fixed or adaptive discovery with explicit fallback provenance."""

    def __init__(
        self,
        index: PersistentResourceIndex,
        *,
        hints: DiscoveryPolicyHints | None = None,
        calibrator: Callable[[float], float] | None = None,
        select_threshold: float = 0.72,
        ask_threshold: float = 0.42,
        margin_threshold: float = 0.08,
        adaptive_path: Sequence[DiscoveryMode | str] = (
            DiscoveryMode.EXPLICIT,
            DiscoveryMode.TOKEN,
            DiscoveryMode.INDEX,
            DiscoveryMode.SEMANTIC,
            DiscoveryMode.HYBRID,
        ),
    ) -> None:
        if not 0.0 <= ask_threshold <= select_threshold <= 1.0:
            raise ValueError("Expected 0 <= ask_threshold <= select_threshold <= 1.")
        self.index = index
        self.hints = hints or DiscoveryPolicyHints()
        self.calibrator = calibrator or (lambda value: value)
        self.select_threshold = select_threshold
        self.ask_threshold = ask_threshold
        self.margin_threshold = margin_threshold
        self.adaptive_path = tuple(DiscoveryMode(mode) for mode in adaptive_path)
        if DiscoveryMode.AUTO in self.adaptive_path or DiscoveryMode.ADAPTIVE in self.adaptive_path:
            raise ValueError("adaptive_path must contain only executable fixed policies.")

    @staticmethod
    def _auto_mode(request: DiscoveryRequest, rows: Sequence[ResourceScore]) -> DiscoveryMode:
        if request.explicit_reference_uris or "!!ref:" in request.query:
            return DiscoveryMode.EXPLICIT
        if max((row.token for row in rows), default=0.0) >= 0.90:
            return DiscoveryMode.TOKEN
        if max((row.index for row in rows), default=0.0) >= 0.75:
            return DiscoveryMode.INDEX
        return DiscoveryMode.SEMANTIC

    @staticmethod
    def _rank(rows: Sequence[ResourceScore], mode: DiscoveryMode) -> tuple[ResourceScore, ...]:
        channel = mode.value
        if mode == DiscoveryMode.EXPLICIT:
            channel = "explicit"
        ranked = sorted(rows, key=lambda row: (-float(getattr(row, channel)), row.uri))
        return tuple(
            replace(
                row,
                selected_score=float(getattr(row, channel)),
                selected_mode=mode.value,
                rank=rank,
            )
            for rank, row in enumerate(ranked, start=1)
        )

    def _confident(self, ranked: Sequence[ResourceScore]) -> tuple[float, float, bool]:
        top = ranked[0].selected_score if ranked else 0.0
        second = ranked[1].selected_score if len(ranked) > 1 else 0.0
        confidence = max(0.0, min(1.0, self.calibrator(top)))
        margin = top - second
        return confidence, margin, confidence >= self.select_threshold and margin >= self.margin_threshold

    def discover(self, request: DiscoveryRequest) -> DiscoveryTrace:
        """Select stable resource IDs and retain every attempted policy stage."""

        rows = self.index.score(request)
        hint = self.hints.resolve(request, self.index.by_uri)
        mode = hint.mode
        if mode == DiscoveryMode.AUTO:
            mode = self._auto_mode(request, rows)
        if mode == DiscoveryMode.ADAPTIVE:
            path = self.adaptive_path
        elif hint.strict:
            path = (mode,)
        else:
            remaining = tuple(value for value in self.adaptive_path if value != mode)
            path = (mode, *remaining)

        executed: list[str] = []
        final_ranked: tuple[ResourceScore, ...] = ()
        confidence = margin = 0.0
        for stage in path:
            executed.append(stage.value)
            ranked = self._rank(rows, stage)
            confidence, margin, confident = self._confident(ranked)
            final_ranked = ranked
            if confident:
                break

        selected = tuple(row.uri for row in final_ranked[: request.top_k] if row.selected_score > 0.0)
        top_resource = self.index.by_uri.get(selected[0]) if selected else None
        if confidence >= self.select_threshold and margin >= self.margin_threshold and selected:
            decision = DiscoveryDecision.SELECT
        elif confidence >= self.ask_threshold or (selected and top_resource and top_resource.side_effect_class != SideEffectClass.NONE):
            decision = DiscoveryDecision.ASK
        else:
            decision = DiscoveryDecision.ABSTAIN
        if request.side_effecting and top_resource is not None and top_resource.side_effect_class in {
            SideEffectClass.WRITE,
            SideEffectClass.DESTRUCTIVE,
        }:
            # Discovery can identify the tool, but cannot authorize a side effect.
            decision = DiscoveryDecision.ASK

        requested = hint.mode.value
        return DiscoveryTrace(
            request=request,
            requested_hint=requested,
            resolved_hint=mode.value,
            strict=hint.strict,
            executed_path=tuple(executed),
            candidates=final_ranked,
            selected_uris=selected,
            confidence=confidence,
            margin=margin,
            decision=decision,
            fallback_count=max(0, len(executed) - 1),
            hint_complied=bool(executed and (requested in {"auto", "adaptive"} or executed[0] == requested)),
            index_fingerprint=self.index.fingerprint.digest,
        )


def resource_uri(kind: str, namespace: str, name: str, version: str) -> str:
    """Construct the canonical prompt-visible identity used by catalog generators."""

    return f"!!ref:{kind}:{namespace}:{name}:{version}!!"
