"""Contracts and metrics for Paper 2.5 end-to-end output validation.

The module deliberately does not retrieve evidence.  It validates frozen
discovery manifests, maps their source spans to cached native-K/V chunks, and
accounts for output quality and physical memory use.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass

from pra_torch.memory import PRACacheEntry, SelectedChunk


@dataclass(frozen=True)
class MaterializationBand:
    """A predeclared set of zero-based decoder layers that may consume memory."""

    name: str
    layers: tuple[int, ...]


@dataclass(frozen=True)
class GenerationCondition:
    """One immutable output condition and its discovery/materialization contract."""

    name: str
    selection: str
    oracle: bool = False
    direct_context: bool = False
    uses_memory: bool = True


MATERIALIZATION_BANDS = (
    MaterializationBand("late_1", (27,)),
    MaterializationBand("late_4", tuple(range(24, 28))),
    MaterializationBand("late_8", tuple(range(20, 28))),
    MaterializationBand("middle_4", tuple(range(12, 16))),
    MaterializationBand("layer_12", (12,)),
    MaterializationBand("topology_sparse", (12, 20, 24, 27)),
    MaterializationBand("all_28", tuple(range(28))),
)

GENERATION_CONDITIONS = (
    GenerationCondition("native_bounded", "none", uses_memory=False),
    GenerationCondition("one_shot", "one_shot"),
    GenerationCondition("graph_sparse", "graph_sparse"),
    GenerationCondition("graph_balanced", "graph_balanced"),
    GenerationCondition("graph_high", "graph_high"),
    GenerationCondition("oracle_evidence", "oracle_evidence", oracle=True),
    GenerationCondition(
        "native_full_context", "full_source", direct_context=True, uses_memory=False
    ),
)


def validate_protocol(
    conditions: Sequence[GenerationCondition] = GENERATION_CONDITIONS,
    bands: Sequence[MaterializationBand] = MATERIALIZATION_BANDS,
    *,
    total_layers: int = 28,
) -> None:
    """Reject ambiguous conditions, missing controls, and malformed layer bands."""
    condition_names = [condition.name for condition in conditions]
    if len(condition_names) != len(set(condition_names)):
        raise ValueError("generation condition names must be unique")
    required = {
        "native_bounded",
        "one_shot",
        "graph_sparse",
        "graph_balanced",
        "graph_high",
        "oracle_evidence",
        "native_full_context",
    }
    if set(condition_names) != required:
        raise ValueError("generation conditions differ from the predeclared C0--C6 matrix")
    for condition in conditions:
        if condition.oracle != (condition.name == "oracle_evidence"):
            raise ValueError("only oracle_evidence may expose oracle-selected spans")
        if condition.direct_context and condition.uses_memory:
            raise ValueError("direct-context controls cannot also activate PRA memory")

    band_names = [band.name for band in bands]
    if len(band_names) != len(set(band_names)):
        raise ValueError("materialization band names must be unique")
    for band in bands:
        if tuple(sorted(set(band.layers))) != band.layers:
            raise ValueError(f"{band.name} layers must be sorted and unique")
        if not band.layers or band.layers[0] < 0 or band.layers[-1] >= total_layers:
            raise ValueError(f"{band.name} is outside the decoder stack")
    all_layers = next((band.layers for band in bands if band.name == "all_28"), ())
    if all_layers != tuple(range(total_layers)):
        raise ValueError("all_28 must activate every intended decoder layer")


def condition_manifest() -> dict[str, object]:
    """Return a JSON-ready record of the exact frozen experiment matrix."""
    validate_protocol()
    return {
        "conditions": [asdict(condition) for condition in GENERATION_CONDITIONS],
        "materialization_bands": [asdict(band) for band in MATERIALIZATION_BANDS],
    }


def normalize_answer(text: str) -> list[str]:
    """Apply the conventional lowercase alphanumeric QA normalization."""
    return re.findall(r"[a-z0-9]+", str(text).casefold())


def deterministic_answer_metrics(prediction: str, reference: str) -> dict[str, float]:
    """Compute deterministic EM, token F1, containment, and normalized accuracy."""
    predicted = normalize_answer(prediction)
    expected = normalize_answer(reference)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    precision = overlap / max(len(predicted), 1)
    recall = overlap / max(len(expected), 1)
    contained = bool(expected) and any(
        predicted[start : start + len(expected)] == expected
        for start in range(max(len(predicted) - len(expected) + 1, 0))
    )
    exact = float(bool(expected) and predicted == expected)
    return {
        "exact_match": exact,
        "token_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "answer_contained": float(contained),
        "normalized_answer_accuracy": max(exact, float(contained)),
    }


def merge_spans(spans: Iterable[Sequence[int]]) -> tuple[tuple[int, int], ...]:
    """Return sorted, clipped-overlap unions without filling genuine source gaps."""
    ordered = sorted((int(start), int(end)) for start, end in spans if int(end) > int(start))
    merged: list[list[int]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return tuple((start, end) for start, end in merged)


def selected_span_metrics(
    selected_spans: Iterable[Sequence[int]],
    evidence_spans: Iterable[Sequence[int]],
    source_tokens: int,
) -> dict[str, float | int]:
    """Measure unique selected/evidence/non-evidence source-token coverage."""
    selected = merge_spans(selected_spans)
    evidence = merge_spans(evidence_spans)
    selected_tokens = sum(end - start for start, end in selected)
    evidence_tokens = sum(end - start for start, end in evidence)
    selected_evidence = sum(
        max(0, min(selected_end, evidence_end) - max(selected_start, evidence_start))
        for selected_start, selected_end in selected
        for evidence_start, evidence_end in evidence
    )
    return {
        "selected_source_tokens": selected_tokens,
        "selected_source_fraction": selected_tokens / max(int(source_tokens), 1),
        "evidence_source_tokens": evidence_tokens,
        "evidence_kv_tokens": selected_evidence,
        "non_evidence_kv_tokens": max(0, selected_tokens - selected_evidence),
        "materialization_density": selected_evidence / max(selected_tokens, 1),
    }


def fixed_chunks_for_spans(
    entry: PRACacheEntry,
    *,
    routing_layer: int,
    selected_spans: Iterable[Sequence[int]],
    selection_name: str,
) -> list[SelectedChunk]:
    """Resolve frozen source spans to exact atomic chunks at the routing layer."""
    spans = merge_spans(selected_spans)
    memory = entry.layer_memory.get(int(routing_layer))
    if memory is None:
        raise ValueError(f"reference has no layer-{routing_layer} native K/V")
    chunks = [
        chunk
        for chunk in memory.chunks
        if any(
            max(chunk.logical_start, start) < min(int(chunk.logical_end), end)
            for start, end in spans
        )
    ]
    if spans and not chunks:
        raise ValueError("frozen selected spans did not map to any cached chunk")
    return [
        SelectedChunk(
            entry=entry,
            chunk=chunk,
            reference_score=0.0,
            chunk_score=0.0,
            layer_id=int(routing_layer),
            reference_rank=1,
            rank_within_reference=rank,
            gist_count=int(chunk.routing_gist.k.shape[0]),
            metadata={
                "selection_policy": "frozen_output_validation",
                "selection_name": selection_name,
                "oracle": selection_name == "oracle_evidence",
            },
        )
        for rank, chunk in enumerate(chunks, start=1)
    ]


def native_kv_accounting(
    *,
    unique_tokens: int,
    materialization_layers: Sequence[int],
    kv_heads: int,
    head_dim: int,
    element_size: int,
) -> dict[str, int | float]:
    """Count unique source tokens separately from physical per-layer K/V states."""
    layers = tuple(sorted(set(map(int, materialization_layers))))
    states = int(unique_tokens) * len(layers)
    byte_count = states * 2 * int(kv_heads) * int(head_dim) * int(element_size)
    return {
        "materialized_unique_tokens": int(unique_tokens),
        "materialization_layer_count": len(layers),
        "native_kv_token_states": states,
        "native_kv_bytes": byte_count,
    }


def exact_generation_join(
    generation_rows: Sequence[Mapping[str, object]],
    discovery_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Join one generation to one frozen discovery row without silent dropping."""
    keys = ("dataset", "example_id", "selection")
    index: dict[tuple[object, ...], Mapping[str, object]] = {}
    for row in discovery_rows:
        key = tuple(row[name] for name in keys)
        if key in index:
            raise ValueError(f"duplicate discovery row: {key}")
        index[key] = row
    joined = []
    seen: set[tuple[object, ...]] = set()
    for row in generation_rows:
        key = tuple(row[name] for name in keys)
        if key in seen:
            raise ValueError(f"duplicate generation row: {key}")
        seen.add(key)
        if key not in index:
            raise ValueError(f"generation row has no frozen discovery row: {key}")
        overlap = set(row).intersection(index[key]) - set(keys)
        if overlap:
            raise ValueError(f"ambiguous generation/discovery fields: {sorted(overlap)}")
        joined.append({**index[key], **row})
    return joined


validate_protocol()
