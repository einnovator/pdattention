"""Controlled associative-chain data for LocalSA and iterative-PRA studies.

The benchmark is deliberately token-level and closed-vocabulary.  Every example
draws a fresh directed chain and fresh distractors, so the answer is determined
by the supplied facts rather than a stable entity mapping.  A reference is one
fact; this makes selected URI sequences directly comparable with the known path
without exposing that path to the router.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, Sequence

import torch


SPECIAL_TOKENS = ("[PAD]", "[BOS]", "[FACT]", "[SEP]", "[Q]", "[A]")
CONTROLLED_PROTOCOL_VERSION = "balanced_terminal_labels_interleaved_decoys_v4"


class ControlledTokenizer:
    """Small deterministic tokenizer whose IDs are stable across all model seeds."""

    def __init__(
        self,
        *,
        entity_count: int = 192,
        relation_count: int = 16,
        filler_count: int = 16,
    ) -> None:
        tokens = [
            *SPECIAL_TOKENS,
            *(f"E{index:03d}" for index in range(entity_count)),
            *(f"R{index:02d}" for index in range(relation_count)),
            *(f"F{index:02d}" for index in range(filler_count)),
        ]
        self.token_to_id = {token: index for index, token in enumerate(tokens)}
        self.id_to_token = tuple(tokens)
        self.pad_token_id = self.token_to_id["[PAD]"]

    @property
    def vocab_size(self) -> int:
        """Return the fixed embedding/output vocabulary width."""
        return len(self.id_to_token)

    def encode(self, text: str) -> list[int]:
        """Map a whitespace-delimited controlled string to token IDs."""
        return [self.token_to_id[token] for token in text.split()]

    def decode(self, token_ids: Iterable[int]) -> str:
        """Map IDs back to a whitespace-delimited controlled string."""
        return " ".join(self.id_to_token[int(token_id)] for token_id in token_ids)


@dataclass(frozen=True)
class ControlledReference:
    """One independently cacheable fact and its graph role."""

    uri: str
    token_ids: tuple[int, ...]
    source: int
    relation: int
    target: int
    is_evidence: bool
    hop: int | None


@dataclass(frozen=True)
class ControlledExample:
    """Full-context training input plus a query-only PRA input and exact path."""

    example_id: str
    depth: int
    full_input_ids: tuple[int, ...]
    query_input_ids: tuple[int, ...]
    answer_id: int
    references: tuple[ControlledReference, ...]
    target_reference_uris: tuple[str, ...]
    evidence_distance: int
    evidence_gap: int
    distractor_count: int
    lexical_overlap: float
    relation_types: int
    branching: int


def _fact_tokens(tokenizer: ControlledTokenizer, source: int, relation: int, target: int) -> list[int]:
    return tokenizer.encode(f"[FACT] E{source:03d} R{relation:02d} E{target:03d} [SEP]")


def make_controlled_example(
    tokenizer: ControlledTokenizer,
    *,
    seed: int,
    depth: int,
    distractor_count: int,
    evidence_gap: int,
    lexical_overlap: float,
    relation_types: int,
    branching: int,
) -> ControlledExample:
    """Generate one randomized path-query example without fixed-label leakage.

    Evidence facts are emitted in reverse hop order.  The final edge maps the
    chain endpoint to one of eight balanced label entities; earlier edges and
    all distractors remain freshly randomized.  This preserves instance-level
    path dependence without turning evaluation into 192-way unseen-token copy.
    This ordering lets a
    causal fact representation absorb its successor when the native receptive
    field reaches far enough, while the query still occurs after every fact.
    Distractors may share a chain source or relation but never reproduce a gold
    ``(source, relation)`` key.
    """
    if depth < 1:
        raise ValueError("depth must be positive.")
    if relation_types < 2:
        raise ValueError("relation_types must be at least two.")
    if not 0.0 <= lexical_overlap <= 1.0:
        raise ValueError("lexical_overlap must be in [0,1].")
    if min(distractor_count, evidence_gap, branching) < 0:
        raise ValueError("distractor_count, evidence_gap, and branching must be non-negative.")

    rng = random.Random(seed)
    entity_count = sum(token.startswith("E") for token in tokenizer.id_to_token)
    relation_count = sum(token.startswith("R") for token in tokenizer.id_to_token)
    label_classes = min(8, entity_count // 2)
    if relation_types >= relation_count:
        raise ValueError("relation_types must leave one reserved label relation.")
    available_entities = list(range(label_classes, entity_count))
    needed_entities = depth + max(distractor_count * 2, 8)
    entities = rng.sample(available_entities, min(needed_entities, len(available_entities)))
    chain = entities[:depth]
    relations = [rng.randrange(relation_types) for _ in range(max(depth - 1, 0))]
    label_relation = relation_count - 1
    answer_entity = rng.randrange(label_classes)
    gold_keys = {
        (chain[index], relations[index]) for index in range(max(depth - 1, 0))
    }
    gold_keys.add((chain[-1], label_relation))
    records: list[tuple[int, int, int, bool, int | None]] = [
        (chain[index], relations[index], chain[index + 1], True, index)
        for index in range(max(depth - 1, 0))
    ]
    records.append((chain[-1], label_relation, answer_entity, True, depth - 1))

    distractor_entities = iter(entities[depth:])
    for index in range(distractor_count):
        # Multiple label-relation decoys prevent a scan-for-R15 shortcut. The
        # correct label is recoverable only after identifying the chain endpoint.
        force_label_decoy = index < max(2, distractor_count // 2)
        overlap = not force_label_decoy and rng.random() < lexical_overlap
        source = chain[rng.randrange(depth)] if overlap else next(distractor_entities)
        relation = label_relation if force_label_decoy else (
            (relations + [label_relation])[rng.randrange(depth)]
            if overlap
            else rng.randrange(relation_types)
        )
        while (source, relation) in gold_keys:
            relation = (relation + 1) % relation_count
        target = (
            rng.randrange(label_classes)
            if relation == label_relation
            else next(distractor_entities, rng.randrange(label_classes, entity_count))
        )
        records.append((source, relation, target, False, None))

    # Explicit branches share a chain source but use a non-gold relation.
    for index in range(branching):
        source_index = index % depth
        source = chain[source_index]
        source_relation = (
            relations[source_index]
            if source_index < len(relations)
            else label_relation
        )
        relation = (source_relation + 1 + index) % relation_count
        while (source, relation) in gold_keys:
            relation = (relation + 1) % relation_count
        target = (
            rng.randrange(label_classes)
            if relation == label_relation
            else rng.randrange(label_classes, entity_count)
        )
        records.append((source, relation, target, False, None))

    evidence = sorted((record for record in records if record[3]), key=lambda row: -int(row[4]))
    distractors = [record for record in records if not record[3]]
    rng.shuffle(distractors)
    evidence_positions = set(rng.sample(range(len(records)), len(evidence)))
    evidence_iterator = iter(evidence)
    distractor_iterator = iter(distractors)
    ordered = [
        next(evidence_iterator) if position in evidence_positions else next(distractor_iterator)
        for position in range(len(records))
    ]
    refs: list[ControlledReference] = []
    evidence_starts: dict[int, int] = {}
    source_tokens = tokenizer.encode("[BOS]")
    filler_ids = [tokenizer.token_to_id[f"F{index:02d}"] for index in range(16)]
    for ref_index, (source, relation, target, is_evidence, hop) in enumerate(ordered):
        uri = f"controlled://{seed}/fact/{ref_index}"
        fact = _fact_tokens(tokenizer, source, relation, target)
        if is_evidence and hop is not None:
            evidence_starts[int(hop)] = len(source_tokens)
        refs.append(
            ControlledReference(
                uri=uri,
                token_ids=tuple(fact),
                source=source,
                relation=relation,
                target=target,
                is_evidence=is_evidence,
                hop=hop,
            )
        )
        source_tokens.extend(fact)
        source_tokens.extend(rng.choices(filler_ids, k=evidence_gap))

    query = tokenizer.encode(
        " ".join(
            [
                "[Q]",
                f"E{chain[0]:03d}",
                *(f"R{relation:02d}" for relation in [*relations, label_relation]),
                "[A]",
            ]
        )
    )
    target_by_hop = {ref.hop: ref.uri for ref in refs if ref.is_evidence}
    return ControlledExample(
        example_id=f"chain-{seed}-d{depth}",
        depth=depth,
        full_input_ids=tuple(source_tokens + query),
        query_input_ids=tuple(tokenizer.encode("[BOS]") + query),
        answer_id=tokenizer.token_to_id[f"E{answer_entity:03d}"],
        references=tuple(refs),
        target_reference_uris=tuple(target_by_hop[index] for index in range(depth)),
        evidence_distance=max(len(source_tokens) - evidence_starts[0], 0),
        evidence_gap=evidence_gap,
        distractor_count=distractor_count,
        lexical_overlap=float(lexical_overlap),
        relation_types=relation_types,
        branching=branching,
    )


def controlled_examples(
    tokenizer: ControlledTokenizer,
    *,
    count: int,
    seed: int,
    depths: Sequence[int] = (1, 2, 3, 4),
    distractors: Sequence[int] = (2, 4, 8),
    evidence_gaps: Sequence[int] = (0, 2, 6),
    lexical_overlaps: Sequence[float] = (0.0, 0.5, 1.0),
    relation_types: Sequence[int] = (4, 8, 15),
    branchings: Sequence[int] = (0, 1, 2),
) -> list[ControlledExample]:
    """Build a deterministic factorial sample spanning all controlled factors."""
    examples = []
    for index in range(count):
        rng = random.Random(seed + index * 104_729)
        examples.append(
            make_controlled_example(
                tokenizer,
                seed=seed + index * 104_729,
                depth=rng.choice(tuple(depths)),
                distractor_count=rng.choice(tuple(distractors)),
                evidence_gap=rng.choice(tuple(evidence_gaps)),
                lexical_overlap=rng.choice(tuple(lexical_overlaps)),
                relation_types=rng.choice(tuple(relation_types)),
                branching=rng.choice(tuple(branchings)),
            )
        )
    return examples


def collate_controlled(
    examples: Sequence[ControlledExample],
    *,
    pad_token_id: int,
    query_only: bool = False,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad inputs and return ``input_ids``, validity mask, and answer IDs."""
    rows = [example.query_input_ids if query_only else example.full_input_ids for example in examples]
    width = max(map(len, rows))
    input_ids = torch.full((len(rows), width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    for row_index, row in enumerate(rows):
        input_ids[row_index, : len(row)] = torch.tensor(row, dtype=torch.long)
        attention_mask[row_index, : len(row)] = 1
    answers = torch.tensor([example.answer_id for example in examples], dtype=torch.long)
    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        answers = answers.to(device)
    return input_ids, attention_mask, answers


def last_valid_logits(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Select ``[B,V]`` logits at each row's final non-padding query token."""
    indices = attention_mask.sum(dim=1).sub(1).clamp_min(0)
    return logits[torch.arange(logits.shape[0], device=logits.device), indices]
