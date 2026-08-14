"""Dataset-truth adapters for natural multi-hop memory-graph experiments.

The adapters retain dataset annotations and serialized source spans.  A later,
explicit mapping step converts those spans to tokenizer and PRA-parent space.
This boundary prevents chunking choices from being mistaken for ground truth.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .chunk_granularity import Span, chunk_spans, span_overlap


@dataclass(frozen=True)
class AnnotatedEvidenceNode:
    """One dataset-provided reasoning/evidence item and its source location."""

    node_id: str
    text_span: tuple[int, int] | None
    dependencies: tuple[str, ...]
    annotation: Mapping[str, object]
    mapping_status: str = "mapped"


@dataclass(frozen=True)
class NaturalReasoningExample:
    """Raw task semantics plus a deterministic memory-source serialization."""

    dataset: str
    example_id: str
    question: str
    answer: str
    question_type: str
    annotated_hops: int
    graph_type: str
    source: str
    nodes: tuple[AnnotatedEvidenceNode, ...]
    raw_annotation: Mapping[str, object]

    @property
    def annotated_edges(self) -> tuple[tuple[str, str], ...]:
        """Return annotation-supported dependency edges, never PRA-derived edges."""
        return tuple(
            (dependency, node.node_id)
            for node in self.nodes
            for dependency in node.dependencies
        )

    @property
    def root_node_ids(self) -> tuple[str, ...]:
        """Return annotated graph entries with no incoming dependency."""
        return tuple(node.node_id for node in self.nodes if not node.dependencies)


@dataclass(frozen=True)
class ParentGraphMapping:
    """One annotation graph mapped post hoc onto a PRA parent partition."""

    parent_spans: tuple[Span, ...]
    node_token_spans: Mapping[str, Span]
    node_parent_groups: Mapping[str, tuple[int, ...]]
    root_parent_ids: tuple[int, ...]
    oracle_parent_ids: tuple[int, ...]
    preserved_edges: tuple[tuple[int, int], ...]
    collapsed_node_edges: tuple[tuple[str, str], ...]
    unmappable_node_edges: tuple[tuple[str, str], ...]


def _normalize_entity(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).casefold()
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"\([^)]*\)", " ", text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _graph_type(node_ids: Sequence[str], edges: Sequence[tuple[str, str]]) -> str:
    if not edges:
        return "independent"
    incoming = {node: 0 for node in node_ids}
    outgoing = {node: 0 for node in node_ids}
    for source, target in edges:
        outgoing[source] += 1
        incoming[target] += 1
    if max(incoming.values()) > 1:
        return "convergent"
    if max(outgoing.values()) > 1:
        return "branching"
    if len(edges) == len(node_ids) - 1:
        return "chain"
    return "disconnected"


def _append_source_part(parts: list[str], text: str) -> tuple[int, int]:
    start = sum(len(part) for part in parts)
    parts.append(text)
    return start, start + len(text)


def parse_musique_row(row: Mapping[str, object]) -> NaturalReasoningExample:
    """Parse one labelled MuSiQue-Ans row and preserve ``#n`` dependencies."""
    decomposition = list(row.get("question_decomposition", []))
    if not decomposition:
        raise ValueError("MuSiQue row has no labelled question decomposition.")
    by_support = {int(step["paragraph_support_idx"]): step for step in decomposition}
    parts: list[str] = []
    support_spans: dict[int, Span] = {}
    for paragraph in row["paragraphs"]:
        index = int(paragraph["idx"])
        parts.append(f"Document {index}: {paragraph['title']}\n")
        span = _append_source_part(parts, str(paragraph["paragraph_text"]))
        parts.append("\n\n")
        if index in by_support:
            support_spans[index] = span
    nodes = []
    for position, step in enumerate(decomposition, start=1):
        dependencies = tuple(
            str(int(match))
            for match in re.findall(r"#(\d+)", str(step["question"]))
            if 1 <= int(match) < position
        )
        support_index = int(step["paragraph_support_idx"])
        nodes.append(
            AnnotatedEvidenceNode(
                node_id=str(position),
                text_span=support_spans.get(support_index),
                dependencies=tuple(dict.fromkeys(dependencies)),
                annotation=dict(step),
                mapping_status=("mapped" if support_index in support_spans else "missing_support"),
            )
        )
    edges = tuple((dependency, node.node_id) for node in nodes for dependency in node.dependencies)
    return NaturalReasoningExample(
        dataset="musique",
        example_id=str(row["id"]),
        question=str(row["question"]),
        answer=str(row.get("answer", "")),
        question_type=f"{len(decomposition)}hop",
        annotated_hops=len(decomposition),
        graph_type=_graph_type([node.node_id for node in nodes], edges),
        source="".join(parts),
        nodes=tuple(nodes),
        raw_annotation={
            "id": row["id"],
            "question_decomposition": decomposition,
            "supporting_paragraph_indices": sorted(by_support),
            "answerable": row.get("answerable"),
        },
    )


def load_musique(path: Path) -> list[NaturalReasoningExample]:
    """Load labelled MuSiQue JSONL without silently accepting test rows."""
    with path.open(encoding="utf-8") as stream:
        return [parse_musique_row(json.loads(line)) for line in stream if line.strip()]


def _best_support_for_evidence(
    evidence: Sequence[object],
    supports: Mapping[tuple[str, int], tuple[str, Span]],
) -> tuple[tuple[str, int] | None, str]:
    subject, relation, object_ = map(str, evidence)
    normalized_subject = _normalize_entity(subject)
    normalized_relation = _normalize_entity(relation)
    normalized_object = _normalize_entity(object_)
    scored: list[tuple[int, tuple[str, int]]] = []
    for key, (sentence, _) in supports.items():
        title = _normalize_entity(key[0])
        text = _normalize_entity(sentence)
        title_match = bool(
            normalized_subject
            and (normalized_subject in title or title in normalized_subject)
        )
        subject_match = bool(normalized_subject and normalized_subject in text)
        object_match = bool(normalized_object and normalized_object in text)
        relation_match = bool(normalized_relation and normalized_relation in text)
        score = 8 * title_match + 4 * subject_match + 2 * object_match + relation_match
        if score:
            scored.append((score, key))
    if not scored:
        return None, "unmapped_no_support_match"
    best_score = max(score for score, _ in scored)
    best = sorted(key for score, key in scored if score == best_score)
    if len(best) != 1:
        return None, "unmapped_ambiguous_support"
    return best[0], "mapped"


def parse_2wiki_row(row: Mapping[str, object]) -> NaturalReasoningExample:
    """Parse one labelled 2Wiki row into evidence nodes and relation joins.

    Evidence triples and supporting facts are dataset annotations.  Dependency
    edges use exact entity-ID joins when available; otherwise exact normalized
    object-to-subject joins are explicitly marked as derived in ``raw_annotation``.
    """
    wanted = {(str(title), int(index)) for title, index in row["supporting_facts"]}
    parts: list[str] = []
    supports: dict[tuple[str, int], tuple[str, Span]] = {}
    for document_index, (title, sentences) in enumerate(row["context"]):
        parts.append(f"Document {document_index}: {title}\n")
        for sentence_index, sentence in enumerate(sentences):
            prefix = f"Sentence {sentence_index}: "
            parts.append(prefix)
            span = _append_source_part(parts, str(sentence))
            parts.append("\n")
            key = (str(title), sentence_index)
            if key in wanted:
                supports[key] = (str(sentence), span)
        parts.append("\n")

    evidences = [tuple(map(str, evidence)) for evidence in row.get("evidences", [])]
    evidence_ids = [tuple(map(str, evidence)) for evidence in row.get("evidences_id", [])]
    joins = evidence_ids if len(evidence_ids) == len(evidences) else evidences
    dependencies: dict[int, list[int]] = {index: [] for index in range(len(evidences))}
    for source, left in enumerate(joins):
        for target, right in enumerate(joins):
            if source != target and _normalize_entity(left[2]) == _normalize_entity(right[0]):
                dependencies[target].append(source)

    nodes = []
    support_assignments = []
    for index, evidence in enumerate(evidences):
        support, status = _best_support_for_evidence(evidence, supports)
        support_assignments.append(
            {"evidence_index": index, "supporting_fact": support, "status": status}
        )
        nodes.append(
            AnnotatedEvidenceNode(
                node_id=str(index),
                text_span=(supports[support][1] if support is not None else None),
                dependencies=tuple(str(value) for value in dependencies[index]),
                annotation={
                    "evidence": evidence,
                    "evidence_id": evidence_ids[index] if index < len(evidence_ids) else None,
                    "supporting_fact": support,
                },
                mapping_status=status,
            )
        )
    edges = tuple((dependency, node.node_id) for node in nodes for dependency in node.dependencies)
    return NaturalReasoningExample(
        dataset="2wikimultihopqa",
        example_id=str(row["_id"]),
        question=str(row["question"]),
        answer=str(row.get("answer", "")),
        question_type=str(row["type"]),
        annotated_hops=max(1, _longest_path_length(nodes)),
        graph_type=_graph_type([node.node_id for node in nodes], edges),
        source="".join(parts),
        nodes=tuple(nodes),
        raw_annotation={
            "id": row["_id"],
            "type": row["type"],
            "supporting_facts": row["supporting_facts"],
            "evidences": row.get("evidences", []),
            "evidences_id": row.get("evidences_id", []),
            "edge_semantics": (
                "dataset_entity_id_exact_join" if evidence_ids else "derived_normalized_entity_join"
            ),
            "support_assignments": support_assignments,
        },
    )


def _longest_path_length(nodes: Sequence[AnnotatedEvidenceNode]) -> int:
    depth: dict[str, int] = {}
    for node in nodes:
        depth[node.node_id] = 1 + max((depth.get(parent, 0) for parent in node.dependencies), default=0)
    return max(depth.values(), default=0)


def load_2wiki(path: Path) -> list[NaturalReasoningExample]:
    """Load a labelled 2Wiki JSON array."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [parse_2wiki_row(row) for row in rows]


def char_spans_to_token_spans(
    offsets: Sequence[Sequence[int]], nodes: Sequence[AnnotatedEvidenceNode]
) -> dict[str, Span]:
    """Map exact serialized character spans through fast-tokenizer offsets."""
    result: dict[str, Span] = {}
    for node in nodes:
        if node.text_span is None:
            continue
        start, end = node.text_span
        token_ids = [
            index
            for index, pair in enumerate(offsets)
            if int(pair[1]) > start and int(pair[0]) < end and int(pair[1]) > int(pair[0])
        ]
        if token_ids:
            result[node.node_id] = (token_ids[0], token_ids[-1] + 1)
    return result


def map_example_to_parents(
    example: NaturalReasoningExample,
    token_count: int,
    node_token_spans: Mapping[str, Span],
    *,
    chunk_size: int,
    overlap: int = 0,
) -> ParentGraphMapping:
    """Map an annotated graph to PRA parents and audit every graph edge."""
    parents = chunk_spans(token_count, chunk_size, overlap)
    groups = {
        node_id: tuple(
            parent_id
            for parent_id, parent_span in enumerate(parents)
            if span_overlap(token_span, parent_span) > 0
        )
        for node_id, token_span in node_token_spans.items()
    }
    preserved: set[tuple[int, int]] = set()
    collapsed, unmappable = [], []
    for source_node, target_node in example.annotated_edges:
        source_group = groups.get(source_node, ())
        target_group = groups.get(target_node, ())
        if not source_group or not target_group:
            unmappable.append((source_node, target_node))
            continue
        new_targets = set(target_group) - set(source_group)
        distinct = {
            (source_parent, target_parent)
            for source_parent in source_group
            for target_parent in new_targets
        }
        if distinct:
            preserved.update(distinct)
        else:
            collapsed.append((source_node, target_node))
    # A long evidence span can overlap several routing chunks.  Entry uses one
    # maximum-overlap representative per annotated root node; all overlaps stay
    # in ``node_parent_groups`` and therefore remain part of oracle evaluation.
    root_parents = tuple(
        dict.fromkeys(
            max(
                groups[node_id],
                key=lambda parent: (
                    span_overlap(node_token_spans[node_id], parents[parent]),
                    -parent,
                ),
            )
            for node_id in example.root_node_ids
            if groups.get(node_id)
        )
    )
    return ParentGraphMapping(
        parent_spans=parents,
        node_token_spans=dict(node_token_spans),
        node_parent_groups=groups,
        root_parent_ids=root_parents,
        oracle_parent_ids=tuple(sorted({parent for group in groups.values() for parent in group})),
        preserved_edges=tuple(sorted(preserved)),
        collapsed_node_edges=tuple(collapsed),
        unmappable_node_edges=tuple(unmappable),
    )


def stable_partition(example_id: str) -> str:
    """Assign identities to validation/test without consulting labels or scores."""
    import hashlib

    return "validation" if hashlib.sha256(example_id.encode()).digest()[0] % 2 == 0 else "test"


def deterministic_stratified_sample(
    examples: Iterable[NaturalReasoningExample],
    per_stratum: int,
    *,
    stratum: str,
) -> list[NaturalReasoningExample]:
    """Select stable identity-sorted samples by hop count or question type."""
    groups: dict[object, list[NaturalReasoningExample]] = {}
    for example in examples:
        key = getattr(example, stratum)
        groups.setdefault(key, []).append(example)
    selected = []
    for key in sorted(groups, key=str):
        ordered = sorted(groups[key], key=lambda item: item.example_id)
        selected.extend(ordered[:per_stratum])
    return selected
