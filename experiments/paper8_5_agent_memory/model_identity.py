"""Observed endpoint identity for controlled agent comparisons."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Mapping
import urllib.request


def validate_ollama_tags(
    payload: Mapping[str, Any],
    *,
    expected_model: str,
    expected_revision: str,
) -> dict[str, Any]:
    matches = [
        row for row in payload.get("models", ())
        if isinstance(row, Mapping)
        and expected_model in {str(row.get("name")), str(row.get("model"))}
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one endpoint model {expected_model!r}; "
            f"observed {len(matches)}"
        )
    row = matches[0]
    digest = str(row.get("digest") or "")
    if digest != expected_revision:
        raise ValueError(
            f"endpoint model digest mismatch: expected {expected_revision}, "
            f"observed {digest or '<missing>'}"
        )
    details = row.get("details") if isinstance(row.get("details"), Mapping) else {}
    return {
        "source": "ollama_api_tags",
        "model": expected_model,
        "revision": digest,
        "format": details.get("format"),
        "family": details.get("family"),
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization_level"),
        "context_length": details.get("context_length"),
    }


def fetch_ollama_model_identity(
    url: str,
    *,
    expected_model: str,
    expected_revision: str,
    timeout_seconds: float = 20,
) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    result = validate_ollama_tags(
        payload,
        expected_model=expected_model,
        expected_revision=expected_revision,
    )
    return {
        **result,
        "url": url,
        "observed_at": datetime.now(timezone.utc).isoformat(),
    }

