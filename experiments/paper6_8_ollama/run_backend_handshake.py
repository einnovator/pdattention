"""Exercise Ollama AUTO negotiation against the pinned llama.cpp E2 receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pra_hf.deployment import PRAEngineResult, PRAWireRequest, PRAWireResource
from pra_ollama import OllamaBackendHandshake, OllamaEngineAdapter


ROOT = Path(__file__).resolve().parents[2]
NATIVE_RECEIPT = (
    ROOT / "docs/papers/shared/results/paper6_7_llamacpp/native_sequence_attach.json"
)
DEFAULT_OUTPUT = (
    ROOT / "docs/papers/shared/results/paper6_8_ollama/backend_handshake.json"
)


class ProbeBackend:
    """Deterministic executor standing in for Ollama's active model runner."""

    def __init__(self, *, mechanisms: tuple[str, ...], stale: bool = False) -> None:
        self.mechanisms = mechanisms
        self.stale = stale
        self.generated = 0
        self.invalidated: list[str] = []

    def negotiate(self, *, model: str, model_fingerprint: str) -> OllamaBackendHandshake:
        del model
        return OllamaBackendHandshake(
            protocol_version="pra-engine/1",
            backend="llama.cpp",
            backend_revision="458681e1d5d",
            model_fingerprint="stale" if self.stale else model_fingerprint,
            model_artifact_digest="sha256:controlled-model",
            integration_level="E2",
            mechanisms=self.mechanisms,
            resource_identity=True,
            tenant_isolation=True,
            request_cleanup=True,
        )

    def generate(
        self, request: PRAWireRequest, handshake: OllamaBackendHandshake
    ) -> PRAEngineResult:
        del request, handshake
        self.generated += 1
        return PRAEngineResult("native")

    def invalidate_model(self, model_fingerprint: str) -> None:
        self.invalidated.append(model_fingerprint)


class ProbeOllama(OllamaEngineAdapter):
    """In-memory Ollama control plane used to isolate negotiation behavior."""

    def __init__(self, backend: ProbeBackend) -> None:
        super().__init__("http://probe.invalid", backend_executor=backend)
        self.calls: list[tuple[str, Any]] = []

    def _request_json(self, path: str, payload=None):
        self.calls.append((path, payload))
        if path == "/api/show":
            return {
                "modified_at": "pinned",
                "details": {"family": "qwen"},
                "model_info": {"backend": "llama.cpp"},
            }
        if path == "/api/chat":
            return {"message": {"content": "selected-text"}}
        return {"done": True}


def request(model: str = "qwen3:0.6b") -> PRAWireRequest:
    return PRAWireRequest(
        model=model,
        messages=({"role": "user", "content": "Question?"},),
        resources=(
            PRAWireResource("doc-1", "memory://doc", text="Selected evidence."),
        ),
        max_new_tokens=4,
    )


def run() -> dict[str, Any]:
    native = json.loads(NATIVE_RECEIPT.read_text(encoding="utf-8"))
    mechanisms = (
        "native_kv",
        "unified_kv_sequence_attach",
        "metadata_only_attach",
        "request_sequence_cleanup",
    )

    valid_backend = ProbeBackend(mechanisms=mechanisms)
    valid = ProbeOllama(valid_backend)
    valid_result = valid.generate(request())

    incomplete_backend = ProbeBackend(mechanisms=("native_kv",))
    incomplete = ProbeOllama(incomplete_backend)
    incomplete_result = incomplete.generate(request())

    stale_backend = ProbeBackend(mechanisms=mechanisms, stale=True)
    stale = ProbeOllama(stale_backend)
    stale_result = stale.generate(request())

    lifecycle_backend = ProbeBackend(mechanisms=mechanisms)
    lifecycle = ProbeOllama(lifecycle_backend)
    lifecycle.generate(request())
    first_fingerprint = lifecycle._model_fingerprint
    lifecycle.generate(request("qwen3:1.7b"))
    switch_invalidated = first_fingerprint in lifecycle_backend.invalidated
    second_fingerprint = lifecycle._model_fingerprint
    lifecycle.unload("qwen3:1.7b")

    rows = [
        {
            "case": "valid_pinned_llamacpp_receipt",
            "resolved_level": valid.capabilities().integration_level.value,
            "result": valid_result.text,
            "native_calls": valid_backend.generated,
        },
        {
            "case": "missing_sequence_attach_mechanisms",
            "resolved_level": incomplete.capabilities().integration_level.value,
            "result": incomplete_result.text,
            "native_calls": incomplete_backend.generated,
        },
        {
            "case": "stale_model_fingerprint",
            "resolved_level": stale.capabilities().integration_level.value,
            "result": stale_result.text,
            "native_calls": stale_backend.generated,
        },
        {
            "case": "model_switch",
            "resolved_level": "E2",
            "old_fingerprint_invalidated": switch_invalidated,
            "fingerprint_changed": first_fingerprint != second_fingerprint,
        },
        {
            "case": "unload",
            "resolved_level": lifecycle.capabilities().integration_level.value,
            "new_fingerprint_invalidated": second_fingerprint
            in lifecycle_backend.invalidated,
        },
    ]
    return {
        "schema_version": "paper6.8-ollama-backend-handshake-v1",
        "evidence_tier": "CONTROLLED_PROTOCOL_PLUS_INHERITED_MODEL_BACKED_MECHANISM",
        "stock_ollama_native_endpoint": False,
        "backend_receipt": {
            "upstream_commit": native["upstream_commit"],
            "runs": native["summary"]["runs"],
            "schedule_matched_exact_logits": native["summary"][
                "schedule_matched_exact_logits"
            ],
            "persistent_decode_exact": native["summary"]["persistent_decode_exact"],
            "absent_request_bounded_1e_2": native["summary"][
                "absent_request_bounded_1e_2"
            ],
            "warm_resource_reuse_exact": native["summary"][
                "warm_resource_reuse_exact"
            ],
            "physical_kv_copy": native["summary"]["physical_kv_copy"],
        },
        "rows": rows,
        "summary": {
            "valid_receipts_upgrade": int(rows[0]["resolved_level"] == "E2"),
            "invalid_receipts_fallback": sum(
                row["resolved_level"] == "E0" for row in rows[1:3]
            ),
            "lifecycle_invalidations": int(switch_invalidated)
            + int(rows[-1]["new_fingerprint_invalidated"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
