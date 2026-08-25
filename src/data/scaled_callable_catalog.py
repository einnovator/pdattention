"""Deterministic callable catalogs for Paper 6.5 discovery scaling.

The 18 frozen target callables remain unchanged. Added callables share either
the target operation or target object, but never both, so they are meaningful
semantic distractors without becoming unlabeled duplicates of the answer.
"""

from __future__ import annotations

import inspect
import random
import re
from dataclasses import dataclass
from typing import Callable

from data.python_tool_ingestion_cases import PAPER6_5_TOOL_CALLABLES


_TARGET_CONCEPTS = {
    "search_user": ("search", "user"),
    "get_user": ("get", "user"),
    "validate_user": ("validate", "user"),
    "update_user": ("update", "user"),
    "notify_user": ("notify", "user"),
    "delete_user": ("delete", "user"),
    "search_document": ("search", "document"),
    "read_document": ("read", "document"),
    "extract_metadata": ("inspect", "document"),
    "update_document": ("update", "document"),
    "export_document": ("export", "document"),
    "search_repository": ("search", "repository"),
    "get_repository": ("get", "repository"),
    "create_issue": ("create", "issue"),
    "update_issue": ("update", "issue"),
    "create_report": ("create", "report"),
    "archive_report": ("archive", "report"),
    "purge_archive": ("delete", "artifact"),
}

_OPERATION_SURFACES = {
    "search": ("search", "find", "locate", "lookup"),
    "get": ("get", "retrieve", "fetch", "load"),
    "validate": ("validate", "verify", "check", "confirm"),
    "update": ("update", "modify", "change", "set"),
    "notify": ("notify", "inform", "alert", "message"),
    "delete": ("delete", "remove", "purge", "erase"),
    "read": ("read", "open", "retrieve", "inspect"),
    "inspect": ("inspect", "extract", "parse", "derive"),
    "export": ("export", "convert", "render", "package"),
    "create": ("create", "generate", "build", "add"),
    "archive": ("archive", "retain", "store", "retire"),
}

_OBJECT_SURFACES = {
    "user": ("user", "account", "profile", "customer"),
    "document": ("document", "file", "text", "article"),
    "repository": ("repository", "repo", "codebase", "project"),
    "issue": ("issue", "ticket", "task", "work_item"),
    "report": ("report", "analysis", "digest", "summary"),
    "artifact": ("artifact", "deliverable", "asset", "package"),
}


@dataclass(frozen=True)
class ScaledCallableSpec:
    """One callable plus known generation provenance for scaling audits."""

    function: Callable[..., object]
    target: bool
    anchor_tool: str
    canonical_operation: str
    canonical_object: str
    confusion_axis: str


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _distractor_callable(
    *,
    index: int,
    anchor_tool: str,
    operation: str,
    object_name: str,
    operation_surface: str,
    object_surface: str,
) -> Callable[..., object]:
    """Create an inert ordinary function whose metadata is fully inspectable."""

    function_name = f"{_slug(operation_surface)}_{_slug(object_surface)}_{index:05d}"
    parameter_name = f"{_slug(object_name)}_id"

    def generated(**kwargs: str) -> dict[str, str]:
        raise NotImplementedError("Synthetic catalog callables are discovery-only fixtures.")

    generated.__name__ = function_name
    generated.__qualname__ = function_name
    generated.__module__ = f"paper6_5.scaled.{_slug(object_name)}"
    generated.__doc__ = (
        f"{operation_surface.title()} a {object_surface} resource in the service catalog.\n\n"
        "Args:\n"
        f"    {parameter_name}: Stable identifier of the {object_surface}.\n"
        "    request_note: Optional request context used for auditing.\n\n"
        "Returns:\n"
        f"    Structured {object_surface} operation result.\n"
    )
    generated.__annotations__ = {
        parameter_name: str,
        "request_note": str,
        "return": dict[str, str],
    }
    generated.__signature__ = inspect.Signature(
        parameters=(
            inspect.Parameter(parameter_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
            inspect.Parameter(
                "request_note",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default="",
                annotation=str,
            ),
        ),
        return_annotation=dict[str, str],
    )
    setattr(generated, "__paper6_5_anchor__", anchor_tool)
    return generated


def generate_scaled_callable_catalog(size: int, *, seed: int) -> tuple[ScaledCallableSpec, ...]:
    """Return 18 targets plus seeded one-facet semantic distractors."""

    if size < len(PAPER6_5_TOOL_CALLABLES):
        raise ValueError(f"Scaled callable catalogs require at least {len(PAPER6_5_TOOL_CALLABLES)} tools.")
    targets = tuple(
        ScaledCallableSpec(
            function=function,
            target=True,
            anchor_tool=function.__name__,
            canonical_operation=_TARGET_CONCEPTS[function.__name__][0],
            canonical_object=_TARGET_CONCEPTS[function.__name__][1],
            confusion_axis="target",
        )
        for function in PAPER6_5_TOOL_CALLABLES
    )
    rng = random.Random(seed)
    operations = tuple(_OPERATION_SURFACES)
    objects = tuple(_OBJECT_SURFACES)
    distractors = []
    for index in range(size - len(targets)):
        anchor = targets[rng.randrange(len(targets))]
        # Alternate the shared facet and reserve every fifth row for a
        # destructive operation, making unsafe exposure measurable at scale.
        share_operation = index % 2 == 0
        if index % 5 == 4 and anchor.canonical_operation != "delete":
            operation = "delete"
            object_name = anchor.canonical_object
            axis = "shared_object_destructive"
        elif share_operation:
            operation = anchor.canonical_operation
            object_name = rng.choice(tuple(value for value in objects if value != anchor.canonical_object))
            axis = "shared_operation"
        else:
            operation = rng.choice(tuple(value for value in operations if value != anchor.canonical_operation))
            object_name = anchor.canonical_object
            axis = "shared_object"
        operation_surface = rng.choice(_OPERATION_SURFACES[operation])
        object_surface = rng.choice(_OBJECT_SURFACES[object_name])
        function = _distractor_callable(
            index=index,
            anchor_tool=anchor.anchor_tool,
            operation=operation,
            object_name=object_name,
            operation_surface=operation_surface,
            object_surface=object_surface,
        )
        distractors.append(ScaledCallableSpec(
            function=function,
            target=False,
            anchor_tool=anchor.anchor_tool,
            canonical_operation=operation,
            canonical_object=object_name,
            confusion_axis=axis,
        ))
    return (*targets, *distractors)
