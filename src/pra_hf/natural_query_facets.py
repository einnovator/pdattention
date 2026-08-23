"""Dataset-derived natural query-facet annotations and evaluation metrics.

The benchmark uses relation chains supplied by 2WikiMultiHopQA and question
decompositions supplied by MuSiQue. It does not treat generated subquestions as
gold and does not claim human inter-annotator agreement. The deterministic
mapper keeps source provenance so that every token decision is auditable.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import torch

from .query_graph_cluster import facet_recovery_metrics


_UNIT_PATTERN = re.compile(r"[\w]+(?:[-'][\w]+)*|[^\w\s]", re.UNICODE)
_CONTENT_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "both", "by", "did", "do",
    "does", "for", "from", "had", "has", "have", "in", "is", "it", "of",
    "on", "or", "that", "the", "their", "then", "to", "was", "were", "with",
}
_ANSWER_WORDS = {"what", "when", "where", "which", "who", "whom", "whose", "how"}
_RELATION_ALIASES = {
    "publication date": {"published", "released", "came", "out", "first", "earlier"},
    "date of birth": {"born", "birth"},
    "place of birth": {"born", "birthplace", "where"},
    "date of death": {"died", "death", "when"},
    "place of death": {"died", "where"},
    "award received": {"award", "won", "received"},
    "educated at": {"attended", "studied", "university", "school"},
    "country of citizenship": {"country", "nationality"},
    "located in the administrative territorial entity": {"located", "where", "region"},
    "spouse": {"spouse", "wife", "husband", "married"},
    "parent": {"parent", "father", "mother"},
    "child": {"child", "son", "daughter"},
}


def _normalize(value: str) -> str:
    value = re.sub(r"['’]s$", "", value.casefold())
    return re.sub(r"[^\w]+", "", value, flags=re.UNICODE)


def _words(value: str) -> set[str]:
    return {
        normalized
        for match in _UNIT_PATTERN.finditer(value)
        if (normalized := _normalize(match.group()))
    }


@dataclass(frozen=True)
class QueryUnitAnnotation:
    """One auditable surface unit in the original natural question."""

    unit_id: int
    char_start: int
    char_end: int
    text: str
    facet_id: int | None
    facet_type: str | None
    shared_global: bool
    confidence: str
    decision: str


@dataclass(frozen=True)
class NaturalFacetAnnotation:
    """A dataset-authored relation/decomposition mapped onto query units."""

    dataset: str
    example_id: str
    split: str
    question: str
    source_schema: str
    source_facets: tuple[str, ...]
    source_dependencies: tuple[tuple[int, int], ...]
    units: tuple[QueryUnitAnnotation, ...]

    def __post_init__(self) -> None:
        if not self.question or len(self.source_facets) < 2:
            raise ValueError("Natural annotations require a multi-facet question.")
        if not self.units or [row.unit_id for row in self.units] != list(range(len(self.units))):
            raise ValueError("Query units must be non-empty and consecutively identified.")
        previous = 0
        for row in self.units:
            if row.char_start < previous or row.char_end <= row.char_start:
                raise ValueError("Query-unit spans must be ordered and non-empty.")
            if self.question[row.char_start : row.char_end] != row.text:
                raise ValueError("Query-unit text must match its source span.")
            if row.facet_id is not None and not 0 <= row.facet_id < len(self.source_facets):
                raise ValueError("Facet identifiers must index source_facets.")
            previous = row.char_end

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "example_id": self.example_id,
            "split": self.split,
            "question": self.question,
            "source_schema": self.source_schema,
            "source_facets": list(self.source_facets),
            "source_dependencies": [list(row) for row in self.source_dependencies],
            "units": [asdict(row) for row in self.units],
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "NaturalFacetAnnotation":
        return cls(
            dataset=str(value["dataset"]),
            example_id=str(value["example_id"]),
            split=str(value["split"]),
            question=str(value["question"]),
            source_schema=str(value["source_schema"]),
            source_facets=tuple(map(str, value["source_facets"])),
            source_dependencies=tuple(tuple(map(int, row)) for row in value["source_dependencies"]),
            units=tuple(QueryUnitAnnotation(**row) for row in value["units"]),
        )


def tokenize_query_units(question: str) -> tuple[tuple[int, int, str], ...]:
    """Split a question into stable word/punctuation units with character spans."""

    return tuple((match.start(), match.end(), match.group()) for match in _UNIT_PATTERN.finditer(question))


def _facet_seed_sets(source_facets: Sequence[str], relations: Sequence[str]) -> list[set[str]]:
    seeds = []
    for index, facet in enumerate(source_facets):
        values = _words(re.sub(r"#\d+", " ", facet).replace(">>", " "))
        relation = relations[index].casefold() if index < len(relations) else ""
        values.update(_RELATION_ALIASES.get(relation, set()))
        seeds.append(values)
    return seeds


def _facet_phrase_sets(source_facets: Sequence[str]) -> list[set[tuple[str, ...]]]:
    """Extract multi-token source phrases that can anchor ambiguous words."""

    output: list[set[tuple[str, ...]]] = []
    for facet in source_facets:
        phrases = set()
        for segment in re.split(r"\s*(?:>>|/)\s*", facet):
            words = tuple(
                normalized
                for match in _UNIT_PATTERN.finditer(segment)
                if (normalized := _normalize(match.group()))
            )
            if len(words) >= 2:
                phrases.add(words)
        output.append(phrases)
    return output


def _map_units(
    question: str,
    source_facets: Sequence[str],
    *,
    relations: Sequence[str] = (),
    answer_to_terminal: bool = True,
) -> tuple[QueryUnitAnnotation, ...]:
    raw = tokenize_query_units(question)
    seeds = _facet_seed_sets(source_facets, relations)
    phrases = _facet_phrase_sets(source_facets)
    assignments: list[int | None] = [None] * len(raw)
    decisions = ["shared_or_global"] * len(raw)
    confidence = ["high"] * len(raw)
    content = []
    normalized_units = [_normalize(text) for _, _, text in raw]
    content = [bool(word and word not in _CONTENT_STOP) for word in normalized_units]
    for facet, candidates in enumerate(phrases):
        for phrase in candidates:
            width = len(phrase)
            for start in range(len(raw) - width + 1):
                if tuple(normalized_units[start : start + width]) == phrase:
                    for index in range(start, start + width):
                        if content[index]:
                            assignments[index] = facet
                            decisions[index] = "exact_source_phrase_match"

    for index, word in enumerate(normalized_units):
        if word in _ANSWER_WORDS and not answer_to_terminal:
            content[index] = False
            decisions[index] = "comparison_global_answer_type"
            continue
        if assignments[index] is not None:
            continue
        if word in _ANSWER_WORDS and answer_to_terminal:
            assignments[index] = len(source_facets) - 1
            decisions[index] = "answer_type_to_terminal_facet"
            continue
        if not content[index]:
            continue
        matches = [facet for facet, values in enumerate(seeds) if word and word in values]
        if len(matches) == 1:
            assignments[index] = matches[0]
            decisions[index] = "unique_source_lexical_match"
        elif len(matches) > 1:
            decisions[index] = "shared_source_lexical_match"

    anchors = [index for index, value in enumerate(assignments) if value is not None]
    for index, is_content in enumerate(content):
        if assignments[index] is not None or not is_content:
            continue
        if decisions[index] == "shared_source_lexical_match":
            continue
        if anchors:
            nearest = min(anchors, key=lambda anchor: (abs(anchor - index), anchor))
            assignments[index] = assignments[nearest]
            decisions[index] = "nearest_source_anchor"
            confidence[index] = "medium"
        else:
            assignments[index] = len(source_facets) - 1
            decisions[index] = "terminal_fallback"
            confidence[index] = "low"

    output = []
    for index, (start, end, text) in enumerate(raw):
        facet = assignments[index]
        output.append(
            QueryUnitAnnotation(
                unit_id=index,
                char_start=start,
                char_end=end,
                text=text,
                facet_id=facet,
                facet_type="relation_chain_step" if facet is not None else None,
                shared_global=facet is None,
                confidence=confidence[index],
                decision=decisions[index],
            )
        )
    return tuple(output)


def annotation_from_musique(record: Mapping, *, split: str) -> NaturalFacetAnnotation:
    """Map MuSiQue's author-supplied decomposition onto the surface question."""

    decomposition = tuple(record.get("question_decomposition") or ())
    source = tuple(str(row["question"]) for row in decomposition)
    dependencies = tuple((index - 1, index) for index in range(1, len(source)))
    return NaturalFacetAnnotation(
        dataset="musique",
        example_id=str(record["id"]),
        split=split,
        question=str(record["question"]),
        source_schema="musique_question_decomposition",
        source_facets=source,
        source_dependencies=dependencies,
        units=_map_units(str(record["question"]), source),
    )


def annotation_from_2wiki(record: Mapping, *, split: str) -> NaturalFacetAnnotation:
    """Map 2Wiki's evidence triples onto the surface question."""

    evidences = tuple(dict.fromkeys(tuple(map(str, row)) for row in (record.get("evidences") or ())))
    is_comparison = str(record.get("type", "")).casefold() == "comparison"
    if is_comparison:
        # Comparison questions expose parallel branches. Group connected
        # triples into those branches instead of mistaking every reasoning
        # edge for an independently expressed query facet.
        parents = list(range(len(evidences)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            left, right = find(left), find(right)
            if left != right:
                parents[max(left, right)] = min(left, right)

        entities = [{_normalize(row[0]), _normalize(row[2])} for row in evidences]
        for left in range(len(evidences)):
            for right in range(left + 1, len(evidences)):
                if entities[left] & entities[right]:
                    union(left, right)
        groups: dict[int, list[tuple[str, ...]]] = {}
        for index, row in enumerate(evidences):
            groups.setdefault(find(index), []).append(row)
        ordered = [groups[key] for key in sorted(groups)]
        source = tuple(" / ".join(" >> ".join(row[:2]) for row in group) for group in ordered)
        relations = tuple(" ".join(row[1] for row in group).casefold() for group in ordered)
        dependencies: tuple[tuple[int, int], ...] = ()
    else:
        source = tuple(" >> ".join(row[:2]) for row in evidences)
        relations = tuple(row[1].casefold() for row in evidences)
        dependencies = tuple((index - 1, index) for index in range(1, len(source)))
    return NaturalFacetAnnotation(
        dataset="2wikimultihopqa",
        example_id=str(record["_id"]),
        split=split,
        question=str(record["question"]),
        source_schema="2wiki_evidence_relation_chain",
        source_facets=source,
        source_dependencies=dependencies,
        units=_map_units(
            str(record["question"]),
            source,
            relations=relations,
            answer_to_terminal=not is_comparison,
        ),
    )


def scorable_labels(annotation: NaturalFacetAnnotation) -> tuple[torch.Tensor, torch.Tensor]:
    """Return target labels and the unit IDs retained for partition metrics."""

    kept = [row for row in annotation.units if row.facet_id is not None]
    if len(kept) < 2:
        raise ValueError("At least two non-global units are required for scoring.")
    return (
        torch.tensor([int(row.facet_id) for row in kept], dtype=torch.long),
        torch.tensor([row.unit_id for row in kept], dtype=torch.long),
    )


def interleaving_statistics(annotation: NaturalFacetAnnotation) -> dict[str, float]:
    """Measure facet switching and non-contiguous surface realization."""

    labels, _ = scorable_labels(annotation)
    switches = int((labels[1:] != labels[:-1]).sum())
    normalized = switches / max(1, labels.numel() - 1)
    non_contiguous = 0
    cover = 0
    for label in torch.unique(labels):
        positions = torch.nonzero(labels == label, as_tuple=False).flatten()
        span = int(positions[-1] - positions[0] + 1)
        cover += span
        non_contiguous += int(span != positions.numel())
    return {
        "facet_switches": float(switches),
        "normalized_switch_rate": float(normalized),
        "minimum_contiguous_cover_ratio": float(cover / max(1, labels.numel())),
        "non_contiguous_facets": float(non_contiguous),
        "has_non_contiguous_facet": float(non_contiguous > 0),
    }


def evaluate_natural_partition(
    predicted_all_units: torch.Tensor,
    annotation: NaturalFacetAnnotation,
) -> dict[str, float]:
    """Evaluate one full-unit prediction under the preregistered exclude-global rule."""

    target, unit_ids = scorable_labels(annotation)
    if predicted_all_units.shape != (len(annotation.units),):
        raise ValueError("Predicted labels must align with every annotated query unit.")
    predicted = predicted_all_units.cpu()[unit_ids]
    metric = facet_recovery_metrics(predicted, target)
    predicted_count = int(torch.unique(predicted).numel())
    target_count = int(torch.unique(target).numel())
    counts = torch.bincount(predicted - predicted.min())
    return {
        "ari": metric.ari,
        "nmi": metric.nmi,
        "pairwise_f1": metric.pairwise_f1,
        "facet_count_abs_error": float(metric.cluster_count_error),
        "over_segmented": float(predicted_count > target_count),
        "under_segmented": float(predicted_count < target_count),
        "singleton_rate": float((counts == 1).sum() / max(1, counts.numel())),
    }


def align_subquestions_to_units(
    annotation: NaturalFacetAnnotation,
    subquestions: Sequence[str],
) -> torch.Tensor:
    """Map generated subquestions to surface units without using gold facet IDs."""

    if not subquestions:
        return torch.zeros(len(annotation.units), dtype=torch.long)
    seeds = [_words(value) for value in subquestions]
    labels = torch.full((len(annotation.units),), -1, dtype=torch.long)
    anchors = []
    for index, unit in enumerate(annotation.units):
        word = _normalize(unit.text)
        matches = [facet for facet, values in enumerate(seeds) if word and word in values]
        if len(matches) == 1:
            labels[index] = matches[0]
            anchors.append(index)
    for index in range(len(annotation.units)):
        if labels[index] >= 0:
            continue
        if anchors:
            nearest = min(anchors, key=lambda anchor: (abs(anchor - index), anchor))
            labels[index] = labels[nearest]
        else:
            labels[index] = 0
    return labels


def annotation_summary(annotations: Iterable[NaturalFacetAnnotation]) -> dict:
    """Aggregate benchmark composition and annotation characteristics."""

    rows = list(annotations)
    by_dataset = {}
    for dataset in sorted({row.dataset for row in rows}):
        subset = [row for row in rows if row.dataset == dataset]
        by_dataset[dataset] = {
            "examples": len(subset),
            "validation": sum(row.split == "validation" for row in subset),
            "test": sum(row.split == "test" for row in subset),
            "mean_facets": sum(len(row.source_facets) for row in subset) / len(subset),
            "shared_token_fraction": sum(
                unit.shared_global for row in subset for unit in row.units
            ) / max(1, sum(len(row.units) for row in subset)),
            "non_contiguous_example_fraction": sum(
                interleaving_statistics(row)["has_non_contiguous_facet"] for row in subset
            ) / len(subset),
        }
    return {
        "examples": len(rows),
        "datasets": by_dataset,
        "human_inter_annotator_agreement": None,
        "annotation_method": "deterministic mapping from dataset-authored decomposition/relation metadata",
        "global_token_rule": "exclude shared/global units from primary partition metrics",
    }
