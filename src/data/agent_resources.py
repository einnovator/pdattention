"""Deterministic opaque tool and skill catalogs for Paper 6.5."""

from __future__ import annotations

import random
import re
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from pra_hf.agent_resources import (
    AgentResource,
    SideEffectClass,
    normalize_text,
    resource_uri,
)


_ACTIONS = (
    ("archive", "place into long term storage"),
    ("compare", "identify differences between"),
    ("convert", "change the representation of"),
    ("create", "make a new"),
    ("delete", "remove permanently"),
    ("export", "send a copy outside"),
    ("inspect", "examine the details of"),
    ("merge", "combine together"),
    ("notify", "send an alert about"),
    ("restore", "recover an earlier"),
    ("search", "find matching"),
    ("summarize", "produce a short account of"),
)

_OBJECTS = (
    ("invoice", "billing document"),
    ("calendar", "meeting schedule"),
    ("repository", "source code project"),
    ("dataset", "collection of records"),
    ("document", "written file"),
    ("image", "visual asset"),
    ("message", "communication item"),
    ("metric", "measurement series"),
    ("report", "analysis result"),
    ("ticket", "work tracking item"),
    ("user", "account holder"),
    ("workflow", "automation sequence"),
)

_SYLLABLES = (
    "bex", "cav", "dun", "fip", "gox", "hex", "jiv", "kaf", "lom", "mep",
    "nuz", "pav", "qir", "rov", "saf", "tib", "vex", "wom", "xil", "zaf",
)


@dataclass(frozen=True)
class CatalogQuery:
    """One selection-only request with hidden evaluation labels."""

    query_id: str
    split: str
    stratum: str
    query: str
    target_uris: tuple[str, ...]
    explicit_reference_uris: tuple[str, ...] = ()
    namespace: str | None = "synthetic"
    expected_decision: str = "select"


@dataclass(frozen=True)
class SyntheticAgentCatalog:
    """Generated resources plus identity-disjoint validation and test requests."""

    resources: tuple[AgentResource, ...]
    queries: tuple[CatalogQuery, ...]
    seed: int

    def split(self, name: str) -> tuple[CatalogQuery, ...]:
        return tuple(query for query in self.queries if query.split == name)


def _opaque_name(index: int, rng: random.Random, kind: str) -> str:
    return f"{kind}_{rng.choice(_SYLLABLES)}_{index:05d}"


def _typo(value: str, rng: random.Random) -> str:
    positions = [index for index, character in enumerate(value) if character.isalpha()]
    if not positions:
        return value + "x"
    index = rng.choice(positions)
    replacement = chr(((ord(value[index].casefold()) - 97 + 7) % 26) + 97)
    return value[:index] + replacement + value[index + 1 :]


def _resource(index: int, rng: random.Random, *, kind: str = "tool") -> AgentResource:
    action, action_paraphrase = _ACTIONS[index % len(_ACTIONS)]
    object_index = (index // len(_ACTIONS)) % len(_OBJECTS)
    object_name, object_paraphrase = _OBJECTS[object_index]
    family = index // (len(_ACTIONS) * len(_OBJECTS))
    name = _opaque_name(index, rng, kind)
    alias = f"{rng.choice(_SYLLABLES)}{index:05d}"
    side_effect = (
        SideEffectClass.DESTRUCTIVE
        if action == "delete"
        else SideEffectClass.WRITE
        if action in {"archive", "convert", "create", "merge", "notify", "restore"}
        else SideEffectClass.READ
    )
    description = f"{action} {object_name} records in service family {family}"
    content = (
        f'{{"name":"{name}","parameters":{{"target":{{"type":"string"}},'
        f'"family":{{"const":{family}}}}}}}'
    )
    return AgentResource(
        uri=resource_uri(kind, "synthetic", name, "v1"),
        kind=kind,
        namespace="synthetic",
        name=name,
        version="v1",
        description=description,
        content=content,
        aliases=(alias,),
        expected_reuse=16,
        side_effect_class=side_effect,
        tenant_id="paper6_5",
        metadata={
            "action": action,
            "action_paraphrase": action_paraphrase,
            "object": object_name,
            "object_paraphrase": object_paraphrase,
            "family": family,
        },
    )


def synthetic_semantic_vector(text: str, dimensions: int = 96) -> tuple[float, ...]:
    """Map held-out paraphrases to catalog concepts before signed hashing.

    This deterministic control stands in for an idealized semantic encoder. It
    does not receive resource identities or relevance labels.
    """

    if dimensions < len(_ACTIONS) + len(_OBJECTS) + 1:
        raise ValueError("Synthetic semantic vectors need action, object, and family slots.")
    normalized = normalize_text(text)
    for canonical, paraphrase in (*_ACTIONS, *_OBJECTS):
        normalized = normalized.replace(normalize_text(paraphrase), canonical)
    tokens = set(normalized.split())
    vector = [0.0] * dimensions
    for index, (canonical, _) in enumerate(_ACTIONS):
        if canonical in tokens:
            vector[index] = 1.0
    object_offset = len(_ACTIONS)
    for index, (canonical, _) in enumerate(_OBJECTS):
        if canonical in tokens:
            vector[object_offset + index] = 1.0
    family = re.search(r"\bfamily\s+(\d+)\b", normalized)
    if family is not None:
        family_offset = len(_ACTIONS) + len(_OBJECTS)
        vector[family_offset + int(family.group(1)) % (dimensions - family_offset)] = 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return tuple(value / norm for value in vector) if norm else tuple(vector)


def _queries_for_resource(
    resource: AgentResource,
    *,
    split: str,
    ordinal: int,
    rng: random.Random,
) -> tuple[CatalogQuery, ...]:
    action = str(resource.metadata["action"])
    action_paraphrase = str(resource.metadata["action_paraphrase"])
    object_name = str(resource.metadata["object"])
    object_paraphrase = str(resource.metadata["object_paraphrase"])
    family = int(resource.metadata["family"])
    alias = resource.aliases[0]
    prefix = f"{split}-{ordinal:05d}"
    return (
        CatalogQuery(
            query_id=f"{prefix}-explicit",
            split=split,
            stratum="explicit_uri",
            query=f"Use {resource.uri} for this request",
            target_uris=(resource.uri,),
            explicit_reference_uris=(resource.uri,),
        ),
        CatalogQuery(
            query_id=f"{prefix}-name",
            split=split,
            stratum="exact_name",
            query=f"Use {resource.name}",
            target_uris=(resource.uri,),
        ),
        CatalogQuery(
            query_id=f"{prefix}-alias",
            split=split,
            stratum="alias",
            query=f"Run {alias}",
            target_uris=(resource.uri,),
        ),
        CatalogQuery(
            query_id=f"{prefix}-typo",
            split=split,
            stratum="typo",
            query=f"Run {_typo(resource.name, rng)}",
            target_uris=(resource.uri,),
        ),
        CatalogQuery(
            query_id=f"{prefix}-semantic",
            split=split,
            stratum="semantic_paraphrase",
            query=(
                f"I need to {action_paraphrase} the {object_paraphrase} "
                f"in service family {family}"
            ),
            target_uris=(resource.uri,),
        ),
        CatalogQuery(
            query_id=f"{prefix}-description",
            split=split,
            stratum="description",
            query=f"Please {action} the {object_name} records in service family {family}",
            target_uris=(resource.uri,),
        ),
    )


def generate_agent_catalog(
    size: int,
    *,
    seed: int,
    validation_identities: int = 8,
    test_identities: int = 24,
    include_skills: bool = False,
) -> SyntheticAgentCatalog:
    """Generate one reproducible catalog and disjoint sampled query identities."""

    if size < 2:
        raise ValueError("Catalog size must be at least two.")
    rng = random.Random(seed)
    resources = tuple(
        _resource(index, rng, kind="skill" if include_skills and index % 5 == 0 else "tool")
        for index in range(size)
    )
    identity_count = min(size, validation_identities + test_identities)
    selected_indices = rng.sample(range(size), identity_count)
    validation_count = min(validation_identities, max(1, identity_count // 3))
    queries: list[CatalogQuery] = []
    for ordinal, index in enumerate(selected_indices):
        split = "validation" if ordinal < validation_count else "test"
        queries.extend(
            _queries_for_resource(resources[index], split=split, ordinal=ordinal, rng=rng)
        )

    # Ambiguous and nonexistent requests evaluate ask/abstain separately from top-1.
    for split, offset in (("validation", 0), ("test", 1)):
        action = _ACTIONS[(seed + offset) % len(_ACTIONS)][0]
        candidates = tuple(
            resource.uri
            for resource in resources
            if resource.metadata.get("action") == action
        )
        queries.append(
            CatalogQuery(
                query_id=f"{split}-ambiguous",
                split=split,
                stratum="ambiguous",
                query=f"Use the {action} operation",
                target_uris=candidates,
                expected_decision="ask",
            )
        )
        queries.append(
            CatalogQuery(
                query_id=f"{split}-nonexistent",
                split=split,
                stratum="nonexistent",
                query=f"Use tool_missing_{seed}_{offset}",
                target_uris=(),
                expected_decision="abstain",
            )
        )
    return SyntheticAgentCatalog(resources=resources, queries=tuple(queries), seed=seed)


def replace_versions(
    resources: Sequence[AgentResource],
    indices: Iterable[int],
) -> tuple[AgentResource, ...]:
    """Return a catalog copy with selected source/version mutations."""

    replacements = set(int(index) for index in indices)
    values = []
    for index, resource in enumerate(resources):
        if index not in replacements:
            values.append(resource)
            continue
        values.append(
            AgentResource(
                **{
                    **resource.__dict__,
                    "uri": resource_uri(
                        resource.kind,
                        resource.namespace,
                        resource.name,
                        "v2",
                    ),
                    "version": "v2",
                    "description": resource.description + "; updated schema",
                }
            )
        )
    return tuple(values)
