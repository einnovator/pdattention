"""Opaque tool-use language for the Paper 6.5 M1 causal model.

The compact grammar keeps tool identity, semantic definition, call arguments,
and execution observations separable. The host binds selected definitions to
request-local slots; the model must read a held-out schema and construct the
corresponding argument rather than transcribe a catalog URI.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Sequence

from pra_hf.agent_resources import AgentResource


@dataclass(frozen=True)
class ToolLanguageExample:
    """One deterministic call-and-observation trajectory for an opaque tool."""

    example_id: str
    resource: AgentResource
    target: str
    observation: str


def schema_code(resource: AgentResource) -> str:
    """Return one of four URI-derived argument schemas absent from the query."""
    digest = hashlib.sha256(resource.uri.encode("utf-8")).digest()
    return ("raw", "num", "id", "val")[digest[0] % 4]


def compact_definition(resource: AgentResource, *, slot: str = "@0") -> str:
    """Serialize the minimum semantics and schema needed to construct a call."""
    return (
        f"D|{slot}|{resource.name}|{schema_code(resource)}|"
        f"{resource.metadata['action']}|"
        f"{resource.metadata['object']}|{resource.metadata['family']}\n"
    )


def compact_query(example: ToolLanguageExample) -> str:
    """Ask for an action without revealing the opaque tool identity."""
    resource = example.resource
    return (
        f"Q|{resource.metadata['action']}|{resource.metadata['object']}|"
        f"{resource.metadata['family']}|{example.target}\nC|"
    )


def formatted_argument(example: ToolLanguageExample) -> str:
    """Apply the definition-only schema to the request's raw target."""
    digits = example.target.removeprefix("x")
    code = schema_code(example.resource)
    if code == "raw":
        return example.target
    if code == "num":
        return digits
    if code == "id":
        return f"id{digits}"
    return f"v{digits}"


def expected_call(example: ToolLanguageExample, *, slot: str = "@0") -> str:
    """Return the exact host-parseable call payload, including its delimiter."""
    return f"{slot}|{schema_code(example.resource)}|{formatted_argument(example)}\n"


def render_definitions(resources: Sequence[AgentResource]) -> str:
    """Bind direct definitions to request-local host slots in visible order."""
    return "".join(
        compact_definition(resource, slot=f"@{index}")
        for index, resource in enumerate(resources)
    )


def continuation_suffix(example: ToolLanguageExample) -> str:
    """Render the authorized tool observation and answer prompt."""
    return f"O|{example.observation}\nA|"


def expected_answer(example: ToolLanguageExample) -> str:
    """Return the observation-grounded continuation target."""
    return f"{example.observation}\n"


def make_tool_examples(
    resources: Sequence[AgentResource],
    *,
    seed: int,
    count: int,
    prefix: str,
) -> tuple[ToolLanguageExample, ...]:
    """Sample deterministic trajectories while balancing resource identities."""
    if not resources:
        raise ValueError("At least one resource is required.")
    rng = random.Random(seed)
    order = list(range(len(resources)))
    rows = []
    for index in range(count):
        if index % len(order) == 0:
            rng.shuffle(order)
        resource = resources[order[index % len(order)]]
        target_number = rng.randrange(100)
        target = f"x{target_number:02d}"
        observation = f"r{(target_number + int(resource.metadata['family'])) % 100:02d}"
        rows.append(
            ToolLanguageExample(
                example_id=f"{prefix}-{index:05d}",
                resource=resource,
                target=target,
                observation=observation,
            )
        )
    return tuple(rows)


def render_supervised_trajectory(
    example: ToolLanguageExample,
    *,
    direct_definitions: Sequence[AgentResource] = (),
) -> tuple[str, tuple[tuple[int, int], ...]]:
    """Render teacher forcing text and character spans carrying supervised loss.

    Definitions and the observation are context. Loss is applied only to the
    generated call and post-observation answer, preserving the causal boundary
    between model output and host execution.
    """
    prefix = render_definitions(direct_definitions)
    prefix += compact_query(example)
    slot_index = next(
        (
            index
            for index, resource in enumerate(direct_definitions)
            if resource.uri == example.resource.uri
        ),
        0,
    )
    call = expected_call(example, slot=f"@{slot_index}")
    middle = continuation_suffix(example)
    answer = expected_answer(example)
    call_start = len(prefix)
    answer_start = call_start + len(call) + len(middle)
    text = prefix + call + middle + answer
    return text, (
        (call_start, call_start + len(call)),
        (answer_start, answer_start + len(answer)),
    )


def parse_call(text: str) -> tuple[str, str, str] | None:
    """Parse the deliberately narrow ``slot|schema|argument`` call grammar."""
    line = text.splitlines()[0] if text else ""
    pieces = line.split("|")
    if len(pieces) != 3 or any(not piece for piece in pieces):
        return None
    return pieces[0], pieces[1], pieces[2]
