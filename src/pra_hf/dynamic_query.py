"""Frozen-model query reconstruction after a memory parent becomes active."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pra_torch.hf.query import token_span_from_offsets

from .query_facets import (
    QueryFacetProvenance,
    QueryFacetSet,
    contextual_window_spans,
)


RECONSTRUCTION_MODES = (
    "query_then_memory",
    "query_separator_memory",
    "memory_then_query",
)
SUPPORT_MODES = ("question_only", "question_and_memory")


@dataclass(frozen=True)
class ReconstructionText:
    """Plain reconstruction content and exact character provenance."""

    content: str
    question_char_span: tuple[int, int]
    memory_char_span: tuple[int, int]
    mode: str


@dataclass(frozen=True)
class ReconstructedQuery:
    """Tokenized Q1 prompt with disjoint question and retrieved-A regions."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    question_span: tuple[int, int]
    memory_span: tuple[int, int]
    mode: str


def reconstruction_text(question: str, memory_text: str, mode: str) -> ReconstructionText:
    """Construct Q+A text without model, router, or oracle-target inputs."""
    question = question.strip()
    memory_text = memory_text.strip()
    if not question or not memory_text:
        raise ValueError("Dynamic reconstruction requires non-empty Q and A text.")
    if mode == "query_then_memory":
        separator = "\n"
        content = question + separator + memory_text
        question_start = 0
        memory_start = len(question) + len(separator)
    elif mode == "query_separator_memory":
        separator = "\n\n[Retrieved memory]\n"
        content = question + separator + memory_text
        question_start = 0
        memory_start = len(question) + len(separator)
    elif mode == "memory_then_query":
        separator = "\n"
        content = memory_text + separator + question
        memory_start = 0
        question_start = len(memory_text) + len(separator)
    else:
        raise ValueError(f"Unsupported reconstruction mode: {mode}")
    return ReconstructionText(
        content=content,
        question_char_span=(question_start, question_start + len(question)),
        memory_char_span=(memory_start, memory_start + len(memory_text)),
        mode=mode,
    )


def render_reconstructed_query(
    tokenizer,
    question: str,
    memory_text: str,
    mode: str,
    *,
    max_tokens: int,
) -> ReconstructedQuery:
    """Render one chat turn and map Q/A characters to contextual token spans."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")
    source = reconstruction_text(question, memory_text, mode)
    if tokenizer.chat_template:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": source.content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        rendered = source.content
    content_start = rendered.rfind(source.content)
    if content_start < 0:
        raise ValueError("Rendered prompt lost reconstructed query content.")
    previous = tokenizer.truncation_side
    try:
        tokenizer.truncation_side = "left"
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_tokens,
        )
    finally:
        tokenizer.truncation_side = previous
    offsets = encoded.pop("offset_mapping")[0].tolist()

    def token_span(char_span: tuple[int, int]) -> tuple[int, int]:
        return token_span_from_offsets(
            offsets,
            content_start + char_span[0],
            content_start + char_span[1],
        )

    return ReconstructedQuery(
        input_ids=encoded.input_ids,
        attention_mask=encoded.attention_mask,
        question_span=token_span(source.question_char_span),
        memory_span=token_span(source.memory_char_span),
        mode=mode,
    )


def build_dynamic_query_facets(
    hidden_states: torch.Tensor,
    *,
    question_span: tuple[int, int],
    memory_span: tuple[int, int],
    support_mode: str,
    window: int = 2,
    stride: int = 1,
    include_global: bool = True,
    native_query: torch.Tensor | None = None,
) -> QueryFacetSet:
    """Pool contextual Q1 facets only from explicitly allowed Q/A regions."""
    if hidden_states.ndim != 2 or hidden_states.shape[0] == 0:
        raise ValueError("Hidden states must have shape [tokens,width].")
    if native_query is not None and (
        native_query.ndim != 3 or native_query.shape[0] != hidden_states.shape[0]
    ):
        raise ValueError("Native query states must align with hidden states.")
    if support_mode == "question_only":
        regions = (("question", question_span),)
    elif support_mode == "question_and_memory":
        regions = (("question", question_span), ("memory_a", memory_span))
    else:
        raise ValueError(f"Unsupported dynamic support mode: {support_mode}")

    rows = [hidden_states[-1]] if include_global else []
    native_rows = [native_query[-1]] if include_global and native_query is not None else []
    provenance = (
        [
            QueryFacetProvenance(
                "global",
                hidden_states.shape[0] - 1,
                hidden_states.shape[0],
                "dynamic_global",
                1,
            )
        ]
        if include_global
        else []
    )
    for family, region in regions:
        for start, end in contextual_window_spans(region, window, stride):
            if end > hidden_states.shape[0]:
                raise ValueError("Dynamic support span exceeds contextual states.")
            rows.append(hidden_states[start:end].mean(dim=0))
            if native_query is not None:
                native_rows.append(native_query[start:end].mean(dim=0))
            provenance.append(
                QueryFacetProvenance(
                    "local", start, end, f"{family}_window_{window}", end - start
                )
            )
    if not rows:
        raise ValueError("Dynamic query requires at least one active facet.")
    return QueryFacetSet(
        hidden=torch.stack(rows),
        native_query=torch.stack(native_rows) if native_query is not None else None,
        provenance=tuple(provenance),
    )
