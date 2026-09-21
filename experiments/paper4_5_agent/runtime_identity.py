"""Fail-closed runtime identity helpers for cross-engine model cohorts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping


_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def observed_source_checkpoint_revision(
    model_source: str | Path,
    *,
    runtime_config: Any | None = None,
) -> str | None:
    """Observe the immutable Hub revision from runtime state or snapshot path.

    A caller-supplied revision is deliberately not accepted as evidence.  The
    loaded configuration's commit hash is preferred; a resolved Hugging Face
    ``snapshots/<sha>`` directory is the local, independently inspectable
    fallback used by MLX and offline engine hosts.
    """

    commit = getattr(runtime_config, "_commit_hash", None)
    if isinstance(commit, str) and _COMMIT.fullmatch(commit.lower()):
        return commit.lower()

    path = Path(model_source).expanduser()
    if path.exists():
        resolved = path.resolve()
        if resolved.parent.name == "snapshots" and _COMMIT.fullmatch(
            resolved.name.lower()
        ):
            return resolved.name.lower()
    return None


def require_source_checkpoint_revision(
    model_source: str | Path,
    expected_revision: str,
    *,
    runtime_config: Any | None = None,
) -> str:
    """Return the observed revision or reject an unbound/drifting runtime."""

    observed = observed_source_checkpoint_revision(
        model_source, runtime_config=runtime_config
    )
    if observed is None:
        raise ValueError(
            "Could not independently observe the loaded source-checkpoint "
            "revision; use a pinned snapshots/<sha> path or a runtime config "
            "that exposes _commit_hash."
        )
    expected = str(expected_revision).lower()
    if observed != expected:
        raise ValueError(
            "Loaded source-checkpoint revision does not match the configured "
            f"revision: observed={observed}, configured={expected}."
        )
    return observed


def validate_health_checkpoint_identity(
    runtime_identity: Mapping[str, Any] | None,
    expected_revision: str,
) -> str:
    """Validate the independently observed revision in an endpoint receipt."""

    if not isinstance(runtime_identity, Mapping):
        raise ValueError(
            "Strict model identity requires an endpoint runtime_identity object."
        )
    observed = runtime_identity.get("observed_source_checkpoint_revision")
    if not isinstance(observed, str) or not _COMMIT.fullmatch(observed.lower()):
        raise ValueError(
            "Strict model identity requires a 40-hex "
            "observed_source_checkpoint_revision."
        )
    expected = str(expected_revision).lower()
    if observed.lower() != expected:
        raise ValueError(
            "Endpoint source-checkpoint revision mismatch: "
            f"observed={observed.lower()}, configured={expected}."
        )
    return observed.lower()
