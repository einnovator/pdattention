"""Qualify real HF sparse-cache inference under request-owned K/V lifecycle."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import platform
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pra_hf.hf_live_kv import HFLiveKVRequestCancelled, HFLiveKVRuntime

from .run_hf_agent_cache_equivalence import (
    _assistant_prompts,
    _max_delta,
    _prefill,
)
from .sparse_gate_common import sparse_causal_plan


def _cpu_cache(cache):
    cloned = copy.copy(cache)
    cloned.layers = []
    for source_layer in cache.layers:
        layer = copy.copy(source_layer)
        for name, value in vars(source_layer).items():
            if torch.is_tensor(value):
                value = value.detach().cpu()
            elif isinstance(value, list):
                value = [
                    item.detach().cpu() if torch.is_tensor(item) else item
                    for item in value
                ]
            setattr(layer, name, value)
        cloned.layers.append(layer)
    return cloned


def _device_cache(cache, device: torch.device):
    cloned = copy.copy(cache)
    cloned.layers = []
    for source_layer in cache.layers:
        layer = copy.copy(source_layer)
        for name, value in vars(source_layer).items():
            if torch.is_tensor(value):
                value = value.to(device)
            elif isinstance(value, list):
                value = [item.to(device) if torch.is_tensor(item) else item for item in value]
            setattr(layer, name, value)
        cloned.layers.append(layer)
    return cloned


def _dump_cache(cache) -> bytes:
    buffer = io.BytesIO()
    torch.save(_cpu_cache(cache), buffer)
    return buffer.getvalue()


def _load_cache(payload: object, device: torch.device):
    if not isinstance(payload, bytes):
        raise TypeError("HF lifecycle offload payload must be bytes.")
    cache = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    return _device_cache(cache, device)


def _fingerprint_cache(cache) -> str:
    digest = hashlib.sha256()
    for layer in cache.layers:
        for name in ("keys", "values", "key_cache", "value_cache"):
            value = getattr(layer, name, None)
            if torch.is_tensor(value):
                digest.update(name.encode("ascii"))
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=args.local_files_only
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device).eval()
    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    prompts = _assistant_prompts(tokenizer, trajectory, args.turn)
    messages = trajectory["messages"]
    assistant_indexes = [
        index for index, message in enumerate(messages)
        if message.get("role") == "assistant"
    ][: args.turn]
    assistant_index = assistant_indexes[-1]
    prompt_ids = prompts[-1]
    tail_tokens = min(args.wire_tail_tokens, max(1, len(prompt_ids) // 4))
    source_ids = prompt_ids[:-tail_tokens]
    wire_tail = prompt_ids[-tail_tokens:]
    source = _prefill(model, source_ids, device)
    source_cache = source.past_key_values
    plan = sparse_causal_plan(
        tokenizer,
        messages[:assistant_index],
        prompt_ids,
        source_tokens=len(source_ids),
        retention_fraction=args.retention_fraction,
    )
    source_fingerprint = _fingerprint_cache(source_cache)
    restored_fingerprints: list[str] = []

    def load_cache(payload: object):
        cache = _load_cache(payload, device)
        restored_fingerprints.append(_fingerprint_cache(cache))
        return cache

    runtime = HFLiveKVRuntime(
        dump=_dump_cache,
        load=load_cache,
    )
    identities = {
        "source_id": "task02-agent-history",
        "tenant_id": "paper4.5",
        "session_id": "task02",
        "generation": 1,
    }
    runtime.register_source(
        identities["source_id"],
        source_cache,
        tenant_id=identities["tenant_id"],
        session_id=identities["session_id"],
        generation=identities["generation"],
    )

    def begin(request_id: str, *, generation: int = 1):
        return runtime.begin_request(
            request_id,
            identities["source_id"],
            plan,
            tenant_id=identities["tenant_id"],
            session_id=identities["session_id"],
            expected_generation=generation,
        )

    candidate = begin("candidate")
    reference = begin("reference")
    two_borrowers = runtime.registry.view(identities["source_id"]).active_request_ids
    try:
        runtime.offload_source(identities["source_id"])
    except RuntimeError as exc:
        two_borrower_offload_error = str(exc)
    else:
        two_borrower_offload_error = ""

    candidate_result = candidate.generate(
        model, wire_tail, max_new_tokens=args.continuation_tokens
    )
    one_borrower = runtime.registry.view(identities["source_id"]).active_request_ids
    try:
        runtime.offload_source(identities["source_id"])
    except RuntimeError as exc:
        one_borrower_offload_error = str(exc)
    else:
        one_borrower_offload_error = ""
    reference_result = reference.generate(
        model, wire_tail, max_new_tokens=args.continuation_tokens
    )

    cancelled = begin("cancelled")
    try:
        cancelled.generate(
            model,
            wire_tail,
            max_new_tokens=args.continuation_tokens,
            cancelled=lambda: True,
        )
    except HFLiveKVRequestCancelled:
        cancellation_observed = True
    else:
        cancellation_observed = False

    try:
        begin("stale", generation=0)
    except RuntimeError as exc:
        stale_error = str(exc)
    else:
        stale_error = ""

    offloaded = runtime.offload_source(identities["source_id"])
    offloaded_view = runtime.registry.view(identities["source_id"])
    restored = begin("restored")
    restored_view = runtime.registry.view(identities["source_id"])
    restored_source_fingerprint = restored_fingerprints[-1]
    restored_result = restored.generate(
        model, wire_tail, max_new_tokens=args.continuation_tokens
    )

    active_at_termination = begin("terminated-active")
    removed = runtime.terminate_session(
        identities["tenant_id"], identities["session_id"]
    )
    try:
        runtime.register_source(
            "replacement",
            source_cache,
            tenant_id=identities["tenant_id"],
            session_id=identities["session_id"],
            generation=2,
        )
    except RuntimeError as exc:
        tombstone_error = str(exc)
    else:
        tombstone_error = ""

    candidate_logits = list(candidate_result.step_logits)
    reference_logits = list(reference_result.step_logits)
    restored_logits = list(restored_result.step_logits)
    checks = {
        "two_concurrent_borrowers": two_borrowers == ("candidate", "reference"),
        "offload_rejected_with_two_borrowers": "while requests borrow" in two_borrower_offload_error,
        "normal_finish_released_exactly_once": (
            candidate.outcome == "finished" and candidate.finish() is False
            and one_borrower == ("reference",)
        ),
        "offload_rejected_with_one_borrower": "while requests borrow" in one_borrower_offload_error,
        "same_subset_token_exact": candidate_result.token_ids == reference_result.token_ids,
        "same_subset_logit_exact": _max_delta(candidate_logits, reference_logits) == 0.0,
        "cooperative_cancel_released_exactly_once": (
            cancellation_observed and cancelled.outcome == "cancelled"
            and cancelled.cancel() is False
        ),
        "stale_generation_rejected": "Stale" in stale_error,
        "idle_offload_reached_non_hot_tier": offloaded_view.tier == "offloaded",
        "restore_reached_hot_tier": restored_view.tier == "hot",
        "restored_source_fingerprint_exact": restored_source_fingerprint == source_fingerprint,
        "restored_generation_token_exact": candidate_result.token_ids == restored_result.token_ids,
        "restored_generation_logit_exact": _max_delta(candidate_logits, restored_logits) == 0.0,
        "termination_cancelled_active_request": (
            active_at_termination.outcome == "cancelled"
            and active_at_termination.cancel() is False
        ),
        "termination_removed_source": removed == 1 and runtime.registry.view(identities["source_id"]) is None,
        "termination_tombstone_rejected_recreation": "terminated" in tombstone_error,
        "original_positions_preserved": candidate_result.source_position_base == len(source_ids),
        "zero_selected_history_reencoding": all(
            row.selected_text_reencoded_tokens == 0
            for row in (candidate_result, reference_result, restored_result)
        ),
    }
    result = {
        "schema_version": "paper4.5.hf-live-kv-lifecycle.v1",
        "probe": "hf_real_model_request_owned_sparse_kv",
        "engine": "transformers-pytorch",
        "model": args.model,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "python_version": platform.python_version(),
        "device": str(device),
        "dtype": args.dtype,
        "trajectory": str(args.trajectory),
        "turn": args.turn,
        "retention_fraction": args.retention_fraction,
        "source_tokens": len(source_ids),
        "wire_suffix_tokens": len(wire_tail),
        "selected_kv_tokens": plan.selected_tokens,
        "realized_retention_fraction": plan.selected_tokens / max(len(source_ids), 1),
        "source_position_base": plan.source_position_base,
        "has_holes": plan.has_holes,
        "physical_kv_copy": candidate_result.physical_kv_copy,
        "selected_text_reencoded_tokens": candidate_result.selected_text_reencoded_tokens,
        "offloaded_payload_type": type(offloaded).__name__,
        "offloaded_payload_bytes": len(offloaded) if isinstance(offloaded, bytes) else None,
        "max_abs_logit_delta_same_subset": _max_delta(candidate_logits, reference_logits),
        "max_abs_logit_delta_after_restore": _max_delta(candidate_logits, restored_logits),
        "checks": checks,
        "runtime_snapshot": runtime.snapshot(),
        "selection_plan": plan.to_dict(),
        "engine_request_callback_integration": True,
        "engine_lifecycle_qualified": all(checks.values()),
        "qualification_blockers": [] if all(checks.values()) else [
            name for name, passed in checks.items() if not passed
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--turn", type=int, default=4)
    parser.add_argument("--retention-fraction", type=float, default=0.9)
    parser.add_argument("--wire-tail-tokens", type=int, default=32)
    parser.add_argument("--continuation-tokens", type=int, default=16)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "engine": result["engine"],
        "model": result["model"],
        "engine_lifecycle_qualified": result["engine_lifecycle_qualified"],
        "qualification_blockers": result["qualification_blockers"],
    }, indent=2))
    raise SystemExit(0 if result["engine_lifecycle_qualified"] else 1)


if __name__ == "__main__":
    main()
