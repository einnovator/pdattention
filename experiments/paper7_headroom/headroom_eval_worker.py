"""Compress official Headroom evaluation cases with released components."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from headroom.cache.compression_store import get_compression_store, reset_compression_store
from headroom.tokenizers import EstimatingTokenCounter
from headroom.transforms import ContentRouter
from headroom.transforms.content_router import ContentRouterConfig


MARKER_RE = re.compile(r"(?:hash=|<<ccr:)([0-9a-fA-F]+)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--without-compaction", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    tokenizer = EstimatingTokenCounter(chars_per_token=4.0)
    rows: list[dict[str, Any]] = []
    for case in payload["rows"]:
        reset_compression_store()
        config = ContentRouterConfig()
        if args.max_items is not None:
            config.smart_crusher_max_items_after_crush = args.max_items
        config.smart_crusher_with_compaction = not args.without_compaction
        router = ContentRouter(config)
        messages = [
            {"role": "user", "content": case["query"]},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": f"call_{case['case_id']}",
                    "type": "function",
                    "function": {"name": "official_eval_context", "arguments": "{}"},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": f"call_{case['case_id']}",
                "content": case["context"],
            },
        ]
        started = time.perf_counter()
        result = router.apply(messages, tokenizer=tokenizer, query=case["query"])
        compression_seconds = time.perf_counter() - started
        compressed = str(result.messages[-1].get("content", ""))
        hashes = list(dict.fromkeys(value.lower() for value in MARKER_RE.findall(compressed)))
        retrieved = []
        retrieval_started = time.perf_counter()
        for hash_key in hashes:
            entry = get_compression_store().retrieve(hash_key, query=case["query"])
            if entry is not None:
                retrieved.append(entry.original_content)
        retrieval_seconds = time.perf_counter() - retrieval_started
        truth = str(case.get("evidence_target", case["ground_truth"]))
        evidence_eligible = int(truth.casefold() in str(case["context"]).casefold())
        rows.append({
            **case,
            "profile": "default" if args.max_items is None else f"max_items_{args.max_items}_ccr",
            "compressed": compressed,
            "retrieved_originals": retrieved,
            "marker_count": len(hashes),
            "all_markers_resolved": int(len(hashes) == len(retrieved)),
            "evidence_eligible": evidence_eligible,
            "evidence_visible_initially": int(truth.casefold() in compressed.casefold()),
            "evidence_visible_after_retrieve": int(
                truth.casefold() in (compressed + "\n" + "\n".join(retrieved)).casefold()
            ),
            "original_tokens": tokenizer.count_text(case["context"]),
            "compressed_tokens": tokenizer.count_text(compressed),
            "retrieved_tokens": sum(tokenizer.count_text(value) for value in retrieved),
            "original_bytes": len(case["context"].encode("utf-8")),
            "compressed_bytes": len(compressed.encode("utf-8")),
            "compression_seconds": compression_seconds,
            "retrieval_seconds": retrieval_seconds,
            "transforms_applied": list(result.transforms_applied),
            "status": "supported",
        })
    args.output.write_text(
        json.dumps({"rows": rows, "loader_failures": payload.get("failures", [])}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} official evaluation rows")


if __name__ == "__main__":
    main()
