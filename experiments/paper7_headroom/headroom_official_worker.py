"""Run released Headroom components in their isolated dependency environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from headroom.cache.compression_store import get_compression_store, reset_compression_store
from headroom.ccr.tool_injection import create_ccr_tool_definition
from headroom.tokenizers import EstimatingTokenCounter
from headroom.transforms import ContentRouter
from headroom.transforms.content_router import ContentRouterConfig


MARKER_RE = re.compile(r"(?:hash=|<<ccr:)([0-9a-fA-F]+)")


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)


def _count(tokenizer: EstimatingTokenCounter, text: str) -> int:
    return int(tokenizer.count_text(text))


def _run_case(
    case: dict[str, Any],
    max_items: int | None,
    *,
    with_compaction: bool,
) -> dict[str, Any]:
    reset_compression_store()
    tokenizer = EstimatingTokenCounter(chars_per_token=4.0)
    config = ContentRouterConfig()
    if max_items is not None:
        config.smart_crusher_max_items_after_crush = max_items
    config.smart_crusher_with_compaction = with_compaction
    router = ContentRouter(config)
    original = _text(case["payload"])
    messages = [
        {"role": "user", "content": str(case["query"])},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": f"call_{case['case_id']}",
                "type": "function",
                "function": {"name": "load_typed_record", "arguments": "{}"},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": f"call_{case['case_id']}",
            "content": original,
        },
    ]
    started = time.perf_counter()
    result = router.apply(
        messages,
        tokenizer=tokenizer,
        query=str(case["query"]),
        target_ratio=None,
    )
    elapsed = time.perf_counter() - started
    compressed = _text(result.messages[-1].get("content", ""))
    hashes = list(dict.fromkeys(match.lower() for match in MARKER_RE.findall(compressed)))
    retrieved: list[str] = []
    retrieval_started = time.perf_counter()
    for hash_key in hashes:
        entry = get_compression_store().retrieve(hash_key, query=str(case["query"]))
        if entry is not None:
            retrieved.append(entry.original_content)
    retrieval_seconds = time.perf_counter() - retrieval_started
    marker = str(case["evidence_marker"])
    tool_definition = create_ccr_tool_definition("openai")
    return {
        "case_id": case["case_id"],
        "partition": case["partition"],
        "case_class": case["case_class"],
        "omission_stratum": case["omission_stratum"],
        "query": case["query"],
        "expected_answer": case["expected_answer"],
        "evidence_marker": marker,
        "expected_action": case["expected_action"],
        "profile": (
            "default"
            if max_items is None and with_compaction
            else f"max_items_{max_items}_{'compact' if with_compaction else 'ccr'}"
        ),
        "max_items_after_crush": max_items if max_items is not None else 15,
        "lossless_first_compaction": int(with_compaction),
        "status": "supported",
        "execution_scope": "official_component_stack",
        "original": original,
        "compressed": compressed,
        "retrieved_originals": retrieved,
        "hashes": hashes,
        "marker_count": len(hashes),
        "all_markers_resolved": int(len(retrieved) == len(hashes)),
        "evidence_visible_initially": int(marker.casefold() in compressed.casefold()),
        "evidence_visible_after_retrieve": int(
            marker.casefold() in (compressed + "\n" + "\n".join(retrieved)).casefold()
        ),
        "backing_contains_evidence": int(marker.casefold() in original.casefold()),
        "original_bytes": len(original.encode("utf-8")),
        "compressed_bytes": len(compressed.encode("utf-8")),
        "retrieved_bytes": sum(len(value.encode("utf-8")) for value in retrieved),
        "original_tokens": _count(tokenizer, original),
        "compressed_tokens": _count(tokenizer, compressed),
        "retrieved_tokens": sum(_count(tokenizer, value) for value in retrieved),
        "retrieval_tool_tokens": _count(tokenizer, json.dumps(tool_definition, sort_keys=True)),
        "compression_seconds": elapsed,
        "retrieval_seconds": retrieval_seconds,
        "transforms_applied": list(result.transforms_applied),
        "warnings": list(result.warnings),
        "compressed_sha256": hashlib.sha256(compressed.encode("utf-8")).hexdigest(),
        "original_sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--without-compaction", action="store_true")
    args = parser.parse_args()
    cases = json.loads(args.input.read_text(encoding="utf-8"))
    rows = [
        _run_case(
            case,
            args.max_items,
            with_compaction=not args.without_compaction,
        )
        for case in cases
    ]
    payload = {
        "headroom_version": importlib.metadata.version("headroom-ai"),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
