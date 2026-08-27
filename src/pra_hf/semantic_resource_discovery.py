"""Model-independent semantic discovery for typed agent resources.

The resolver keeps lexical, concept, metadata, and compact-embedding evidence
separate. It never calls a generative model and can therefore run beside a
remote or opaque model provider. Model-native Q/K remains a later optional
fallback rather than an implicit dependency of this module.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from pra_hf.agent_resources import (
    AgentResource,
    DiscoveryRequest,
    PersistentResourceIndex,
    normalize_text,
    terms,
)


@dataclass(frozen=True)
class ConceptExpansion:
    """One auditable surface-to-concept mapping used at query time."""

    surface: str
    canonical: str
    kind: str
    language: str
    source: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in {"operation", "object"}:
            raise ValueError("Concept kind must be operation or object.")
        if not 0.0 < self.weight <= 1.0:
            raise ValueError("Concept weights must be in (0, 1].")


class CanonicalConceptMap:
    """Layered, domain-aware lexical mappings with retained provenance."""

    def __init__(self, expansions: Iterable[ConceptExpansion]) -> None:
        self.expansions = tuple(expansions)
        if not self.expansions:
            raise ValueError("At least one canonical concept expansion is required.")

    @classmethod
    def from_json(cls, path: str | Path) -> "CanonicalConceptMap":
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(ConceptExpansion(**row) for row in rows["expansions"])

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": "1.0", "expansions": [asdict(row) for row in self.expansions]}

    def match(self, text: str, *, language: str) -> tuple[ConceptExpansion, ...]:
        """Return longest-surface matches for one declared language plus English aliases."""

        normalized = f" {normalize_text(text)} "
        matches = []
        for row in self.expansions:
            if row.language not in {language, "all"}:
                continue
            surface = normalize_text(row.surface)
            if surface and f" {surface} " in normalized:
                matches.append(row)
        kind_order = {"operation": 0, "object": 1}
        matches.sort(
            key=lambda row: (
                kind_order[row.kind],
                -len(terms(row.surface)),
                row.canonical,
                row.surface,
            )
        )
        deduplicated = {}
        for row in matches:
            deduplicated.setdefault((row.kind, row.canonical), row)
        return tuple(deduplicated.values())

    def concepts(self, text: str, *, language: str) -> dict[str, dict[str, float]]:
        values: dict[str, dict[str, float]] = {"operation": {}, "object": {}}
        for row in self.match(text, language=language):
            previous = values[row.kind].get(row.canonical, 0.0)
            values[row.kind][row.canonical] = max(previous, row.weight)
        return values

    def expanded_text(self, text: str, *, language: str) -> tuple[str, tuple[ConceptExpansion, ...]]:
        matched = self.match(text, language=language)
        suffix = " ".join(row.canonical for row in matched)
        return " ".join(value for value in (text, suffix) if value), matched


@dataclass(frozen=True)
class ToolSemanticCard:
    """Deterministic registration-time representation of one tool capability."""

    uri: str
    name: str
    description: str
    operation: str
    objects: tuple[str, ...]
    tags: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    effect: str

    @classmethod
    def from_resource(cls, resource: AgentResource) -> "ToolSemanticCard":
        schema = json.loads(resource.content) if resource.content else {}
        parameters = schema.get("parameters", {}).get("properties", {})
        return cls(
            uri=resource.uri,
            name=resource.name,
            description=resource.description,
            operation=resource.operation_kind or "unknown",
            objects=tuple(sorted(resource.object_types)),
            tags=tuple(sorted(resource.tags | resource.auto_tags)),
            inputs=tuple(sorted(str(value) for value in parameters)),
            outputs=tuple(sorted(resource.produces)),
            effect=resource.side_effect_class.value,
        )

    @property
    def description_text(self) -> str:
        return self.description

    @property
    def name_description_text(self) -> str:
        return f"{self.name.replace('_', ' ')}. {self.description}"

    @property
    def structured_text(self) -> str:
        return "\n".join(
            (
                f"Purpose: {self.description}",
                f"Operation: {self.operation}",
                f"Object: {' '.join(self.objects)}",
                f"Inputs: {' '.join(self.inputs)}",
                f"Outputs: {' '.join(self.outputs)}",
                f"Effect: {self.effect}",
                f"Tags: {' '.join(self.tags)}",
            )
        )

    @property
    def vectors(self) -> tuple[str, ...]:
        """E3 multi-vector fields kept separate for max-similarity scoring."""

        return (
            self.name_description_text,
            f"operation {self.operation}; object {' '.join(self.objects)}; tags {' '.join(self.tags)}",
            f"inputs {' '.join(self.inputs)}; outputs {' '.join(self.outputs)}; effect {self.effect}",
        )


class CompactEmbeddingEncoder:
    """Small local Transformers encoder with normalized mean pooling."""

    def __init__(
        self,
        model_id: str,
        *,
        revision: str,
        device: torch.device | str = "cpu",
        query_prefix: str = "",
        pooling: str = "mean",
        local_files_only: bool = True,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        started = time.perf_counter()
        self.model_id = model_id
        self.revision = revision
        self.device = torch.device(device)
        self.query_prefix = query_prefix
        if pooling not in {"mean", "cls"}:
            raise ValueError("Compact embedding pooling must be mean or cls.")
        self.pooling = pooling
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
        ).to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.cold_load_seconds = time.perf_counter() - started

    @property
    def dimensions(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def parameter_bytes(self) -> int:
        return sum(parameter.numel() * parameter.element_size() for parameter in self.model.parameters())

    def encode(
        self,
        texts: Sequence[str],
        *,
        query: bool = False,
        batch_size: int = 32,
        max_length: int = 256,
    ) -> torch.Tensor:
        """Encode text to unit vectors on CPU; registration and query timing stay separable."""

        if not texts:
            return torch.empty((0, self.dimensions), dtype=torch.float32)
        values = [f"{self.query_prefix}{text}" if query else text for text in texts]
        rows = []
        for start in range(0, len(values), batch_size):
            tokens = self.tokenizer(
                values[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                hidden = self.model(**tokens, return_dict=True).last_hidden_state
            if self.pooling == "cls":
                pooled = hidden[:, 0]
            else:
                mask = tokens.attention_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            rows.append(F.normalize(pooled.float(), dim=-1).cpu())
        return torch.cat(rows, dim=0)


@dataclass(frozen=True)
class ExternalScoreRow:
    """Aligned evidence channels for one candidate tool."""

    uri: str
    token: float
    bm25: float
    dictionary: float
    tags: float
    embedding: float


class ExternalSemanticIndex:
    """Persistent lexical/concept/card index with optional compact embeddings."""

    def __init__(
        self,
        resources: Sequence[AgentResource],
        concepts: CanonicalConceptMap,
        *,
        cards: Sequence[ToolSemanticCard] | None = None,
        embeddings: torch.Tensor | None = None,
        multi_embeddings: torch.Tensor | None = None,
    ) -> None:
        self.resources = tuple(resources)
        self.cards = tuple(cards or (ToolSemanticCard.from_resource(row) for row in resources))
        if tuple(card.uri for card in self.cards) != tuple(resource.uri for resource in self.resources):
            raise ValueError("Semantic cards must preserve resource order and identity.")
        self.concepts = concepts
        self.lexical = PersistentResourceIndex(self.resources)
        self.embeddings = embeddings
        self.multi_embeddings = multi_embeddings
        if embeddings is not None and embeddings.shape[0] != len(self.resources):
            raise ValueError("Embedding rows must align with resources.")
        if multi_embeddings is not None and multi_embeddings.shape[:2] != (len(self.resources), 3):
            raise ValueError("Multi-vector embeddings must have shape [resources, 3, dimensions].")

    @property
    def estimated_lexical_bytes(self) -> int:
        return self.lexical.estimated_bytes

    @property
    def embedding_bytes(self) -> int:
        values = self.multi_embeddings if self.multi_embeddings is not None else self.embeddings
        return int(values.numel() * values.element_size()) if values is not None else 0

    @staticmethod
    def _concept_score(
        query: Mapping[str, Mapping[str, float]],
        card: ToolSemanticCard,
    ) -> tuple[float, float]:
        operation = query["operation"].get(card.operation, 0.0)
        objects = max((query["object"].get(value, 0.0) for value in card.objects), default=0.0)
        dictionary = 0.65 * operation + 0.35 * objects
        query_concepts = set(query["operation"]) | set(query["object"])
        card_terms = {card.operation, *card.objects, *card.tags}
        tags = len(query_concepts & card_terms) / max(len(query_concepts), 1)
        return dictionary, tags

    def score(
        self,
        query: str,
        *,
        context: str = "",
        language: str = "en",
        query_embedding: torch.Tensor | None = None,
    ) -> tuple[ExternalScoreRow, ...]:
        """Score one query while giving every channel the same bounded context."""

        full_query = "\n".join(value for value in (context, query) if value)
        expanded, _ = self.concepts.expanded_text(full_query, language=language)
        concept_values = self.concepts.concepts(full_query, language=language)
        request = DiscoveryRequest(query=full_query, tenant_id="paper6_5", top_k=len(self.resources))
        expanded_request = DiscoveryRequest(query=expanded, tenant_id="paper6_5", top_k=len(self.resources))
        lexical = {row.uri: row for row in self.lexical.score(request, channels=("token", "index"))}
        expanded_lexical = {
            row.uri: row for row in self.lexical.score(expanded_request, channels=("token", "index"))
        }
        embedding_scores = torch.zeros(len(self.resources))
        if query_embedding is not None:
            query_embedding = F.normalize(query_embedding.float().reshape(1, -1), dim=-1)
            if self.multi_embeddings is not None:
                embedding_scores = torch.einsum(
                    "rd,rvd->rv",
                    query_embedding.expand(len(self.resources), -1),
                    self.multi_embeddings.float(),
                ).max(dim=1).values
            elif self.embeddings is not None:
                embedding_scores = self.embeddings.float() @ query_embedding[0]
            embedding_scores = embedding_scores.clamp(-1.0, 1.0)
        rows = []
        for index, card in enumerate(self.cards):
            base = lexical.get(card.uri)
            enriched = expanded_lexical.get(card.uri)
            dictionary, tag_score = self._concept_score(concept_values, card)
            rows.append(
                ExternalScoreRow(
                    uri=card.uri,
                    token=float(base.token if base else 0.0),
                    bm25=float(base.index if base else 0.0),
                    dictionary=max(dictionary, float(enriched.index if enriched else 0.0)),
                    tags=tag_score,
                    embedding=float(embedding_scores[index]),
                )
            )
        return tuple(rows)


def token_overlap(query: str, resource: AgentResource) -> float:
    """Continuous lexical-hardness measure required by the M6.5 protocol."""

    query_terms = set(terms(query))
    return len(query_terms & set(terms(resource.search_text))) / max(len(query_terms), 1)
