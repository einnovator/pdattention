"""Qualify real HF sparse-cache inference under request-owned K/V lifecycle."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import io
import json
import platform
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pra_hf.hf_live_kv import (
    HFLiveKVRequest,
    HFLiveKVRequestCancelled,
    HFLiveKVRuntime,
)
from pra_hf.agent_executor import split_generation_prompt

from .run_hf_agent_cache_equivalence import _max_delta
from .frozen_agent_plan import (
    frozen_live_kv_geometry,
    load_frozen_agent_decisions,
)
from .sparse_gate_common import sparse_causal_plan


def _last_logit_kwargs(model) -> dict[str, int]:
    forward = getattr(model, "forward", model)
    try:
        parameters = inspect.signature(forward).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "logits_to_keep" in parameters:
        return {"logits_to_keep": 1}
    if "num_logits_to_keep" in parameters:
        return {"num_logits_to_keep": 1}
    return {}


def _require_live_kv_api(request_class=HFLiveKVRequest) -> str:
    """Fail before an expensive prefill when an older PRA package shadows src/."""

    parameters = inspect.signature(request_class.generate).parameters
    module_path = inspect.getsourcefile(request_class) or inspect.getfile(request_class)
    if "materialized_history" not in parameters:
        raise RuntimeError(
            "Incompatible pra_hf live-K/V API: HFLiveKVRequest.generate lacks "
            f"materialized_history (loaded from {module_path}). Put this "
            "worktree's src directory first on PYTHONPATH."
        )
    return str(Path(module_path).resolve())


@torch.inference_mode()
def _bounded_extend(
    model,
    cache,
    ids: list[int],
    device: torch.device,
    *,
    start_position: int,
    step_size: int,
):
    if step_size < 1:
        raise ValueError("prefill step size must be positive")
    if not ids:
        raise ValueError("HF lifecycle cache extension requires at least one token")
    output = None
    kwargs = _last_logit_kwargs(model)
    for offset in range(0, len(ids), step_size):
        step = ids[offset : offset + step_size]
        positions = torch.arange(
            start_position + offset,
            start_position + offset + len(step),
            device=device,
        )
        output = model(
            input_ids=torch.tensor([step], dtype=torch.long, device=device),
            past_key_values=cache,
            cache_position=positions,
            position_ids=positions.unsqueeze(0),
            use_cache=True,
            return_dict=True,
            **kwargs,
        )
        cache = output.past_key_values
    assert output is not None
    return output, cache


@torch.inference_mode()
def _bounded_prefill(model, ids: list[int], device: torch.device, *, step_size: int):
    from transformers import DynamicCache

    if not ids:
        raise ValueError("HF lifecycle prefill requires at least one token")
    return _bounded_extend(
        model,
        DynamicCache(),
        ids,
        device,
        start_position=0,
        step_size=step_size,
    )


@torch.inference_mode()
def _ordinary_generate(
    model,
    source_ids: list[int],
    wire_tail: list[int],
    device: torch.device,
    *,
    max_new_tokens: int,
    prefill_step_size: int,
) -> tuple[tuple[int, ...], list[torch.Tensor]]:
    _source_output, cache = _bounded_prefill(
        model, source_ids, device, step_size=prefill_step_size
    )
    output, cache = _bounded_extend(
        model,
        cache,
        wire_tail,
        device,
        start_position=len(source_ids),
        step_size=prefill_step_size,
    )
    logits = output.logits
    generated: list[int] = []
    step_logits: list[torch.Tensor] = []
    for step in range(max_new_tokens):
        current = logits[:, -1, :]
        step_logits.append(current.detach().float().cpu())
        token = int(torch.argmax(current, dim=-1).item())
        generated.append(token)
        if step + 1 == max_new_tokens:
            break
        position = torch.tensor(
            [len(source_ids) + len(wire_tail) + step],
            dtype=torch.long,
            device=device,
        )
        output = model(
            input_ids=torch.tensor([[token]], dtype=torch.long, device=device),
            past_key_values=cache,
            cache_position=position,
            position_ids=position.unsqueeze(0),
            use_cache=True,
            return_dict=True,
            **_last_logit_kwargs(model),
        )
        logits = output.logits
        cache = output.past_key_values
    return tuple(generated), step_logits


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
    hf_live_kv_module = _require_live_kv_api()
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
    frozen_decision = None
    if args.request_replay or args.selection_fixture:
        if not args.request_replay or not args.selection_fixture:
            raise ValueError(
                "frozen replay requires both --request-replay and --selection-fixture"
            )
        decisions = load_frozen_agent_decisions(
            args.request_replay, args.selection_fixture
        )
        if args.request_index < 1 or args.request_index > len(decisions):
            raise ValueError("--request-index is outside the frozen replay")
        frozen_decision = decisions[args.request_index - 1]
        geometry = frozen_live_kv_geometry(
            tokenizer,
            frozen_decision,
            full_retention=args.frozen_full_retention,
        )
        prompt_ids = list(geometry.prompt_ids)
        source_ids = list(geometry.source_ids)
        wire_tail = list(geometry.wire_tail_ids)
        plan = geometry.plan
        materialized_history = tuple(
            (span.token_ids, span.position_start)
            for span in geometry.materialized_history_spans
        )
    else:
        if args.trajectory is None:
            raise ValueError(
                "provide --trajectory or the frozen request/selection pair"
            )
        trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
        messages = trajectory["messages"]
        assistant_indexes = [
            index for index, message in enumerate(messages)
            if message.get("role") == "assistant"
        ][: args.turn]
        assistant_index = assistant_indexes[-1]
        prompt_ids, source_ids, wire_tail = split_generation_prompt(
            tokenizer,
            messages[:assistant_index],
        )
        plan = sparse_causal_plan(
            tokenizer,
            messages[:assistant_index],
            prompt_ids,
            source_tokens=len(source_ids),
            retention_fraction=args.retention_fraction,
        )
        materialized_history = ()
    ordinary_full_tokens = None
    ordinary_full_logits = None
    if frozen_decision and args.frozen_full_retention:
        ordinary_full_tokens, ordinary_full_logits = _ordinary_generate(
            model,
            source_ids,
            wire_tail,
            device,
            max_new_tokens=args.continuation_tokens,
            prefill_step_size=args.prefill_step_size,
        )
    _source_output, source_cache = _bounded_prefill(
        model,
        source_ids,
        device,
        step_size=args.prefill_step_size,
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
        model,
        wire_tail,
        max_new_tokens=args.continuation_tokens,
        materialized_history=materialized_history,
    )
    one_borrower = runtime.registry.view(identities["source_id"]).active_request_ids
    try:
        runtime.offload_source(identities["source_id"])
    except RuntimeError as exc:
        one_borrower_offload_error = str(exc)
    else:
        one_borrower_offload_error = ""
    reference_result = reference.generate(
        model,
        wire_tail,
        max_new_tokens=args.continuation_tokens,
        materialized_history=materialized_history,
    )

    cancelled = begin("cancelled")
    try:
        cancelled.generate(
            model,
            wire_tail,
            max_new_tokens=args.continuation_tokens,
            cancelled=lambda: True,
            materialized_history=materialized_history,
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
        model,
        wire_tail,
        max_new_tokens=args.continuation_tokens,
        materialized_history=materialized_history,
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
        "ordinary_full_engine_oracle_token_exact": (
            ordinary_full_tokens is None
            or candidate_result.token_ids == ordinary_full_tokens
        ),
        "ordinary_full_engine_oracle_logit_within_tolerance": (
            ordinary_full_logits is None
            or _max_delta(candidate_logits, ordinary_full_logits)
            <= args.max_abs_logit_delta
        ),
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
        "selected_history_reencoding_equals_explicit_materialization": all(
            row.selected_text_reencoded_tokens
            == sum(len(tokens) for tokens, _position in materialized_history)
            for row in (candidate_result, reference_result, restored_result)
        ),
    }
    result = {
        "schema_version": "paper4.5.hf-live-kv-lifecycle.v4",
        "probe": "hf_real_model_request_owned_sparse_kv",
        "engine": "transformers-pytorch",
        "hf_live_kv_module": hf_live_kv_module,
        "model": args.model,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "python_version": platform.python_version(),
        "device": str(device),
        "dtype": args.dtype,
        "trajectory": str(args.trajectory) if args.trajectory else None,
        "request_replay": str(args.request_replay) if args.request_replay else None,
        "selection_fixture": (
            str(args.selection_fixture) if args.selection_fixture else None
        ),
        "request_index": (
            frozen_decision.request_index if frozen_decision else None
        ),
        "request_input_sha256": (
            frozen_decision.request_input_sha256 if frozen_decision else None
        ),
        "source_policy": (
            frozen_decision.source_policy if frozen_decision else None
        ),
        "turn": args.turn if frozen_decision is None else None,
        "retention_fraction": (
            (
                plan.selected_tokens
                + sum(len(tokens) for tokens, _position in materialized_history)
                + len(wire_tail)
            ) / max(len(prompt_ids), 1)
            if frozen_decision else args.retention_fraction
        ),
        "source_tokens": len(source_ids),
        "wire_suffix_tokens": len(wire_tail),
        "prefill_step_size": args.prefill_step_size,
        "selected_kv_tokens": plan.selected_tokens,
        "materialized_history_tokens": sum(
            len(tokens) for tokens, _position in materialized_history
        ),
        "realized_retention_fraction": (
            (
                plan.selected_tokens
                + sum(len(tokens) for tokens, _position in materialized_history)
                + len(wire_tail)
            ) / max(len(prompt_ids), 1)
            if frozen_decision
            else plan.selected_tokens / max(len(source_ids), 1)
        ),
        "historical_kv_retention_fraction": (
            plan.selected_tokens / max(len(source_ids), 1)
        ),
        "source_position_base": plan.source_position_base,
        "has_holes": plan.has_holes,
        "physical_kv_copy": candidate_result.physical_kv_copy,
        "interval_pack_bytes": candidate_result.interval_pack_bytes,
        "transient_attention_bytes": candidate_result.transient_attention_bytes,
        "transient_kv_copy_bytes": candidate_result.transient_kv_copy_bytes,
        "max_transient_kv_tile_bytes": candidate_result.max_transient_kv_tile_bytes,
        "request_tail_copy_bytes": candidate.selection.request_tail_copy_bytes,
        "fused_attention_calls": candidate.selection.fused_attention_calls,
        "streaming_attention_calls": candidate.selection.streaming_attention_calls,
        "cuda_attention_backend": (
            "triton_fused"
            if candidate.selection.fused_attention_calls > 0
            and candidate.selection.streaming_attention_calls == 0
            else "streaming_segmented"
            if candidate.selection.streaming_attention_calls > 0
            and candidate.selection.fused_attention_calls == 0
            else "mixed_or_unused"
        ),
        "selected_text_reencoded_tokens": candidate_result.selected_text_reencoded_tokens,
        "offloaded_payload_type": type(offloaded).__name__,
        "offloaded_payload_bytes": len(offloaded) if isinstance(offloaded, bytes) else None,
        "max_abs_logit_delta_same_subset": _max_delta(candidate_logits, reference_logits),
        "ordinary_full_engine_oracle_token_ids": (
            list(ordinary_full_tokens)
            if ordinary_full_tokens is not None
            else None
        ),
        "ordinary_full_engine_oracle_mode": (
            "independent_resident_prefix_then_wire_tail"
            if ordinary_full_tokens is not None
            else None
        ),
        "max_abs_logit_delta_ordinary_full_engine_oracle": (
            _max_delta(candidate_logits, ordinary_full_logits)
            if ordinary_full_logits is not None
            else None
        ),
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
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--request-replay", type=Path)
    parser.add_argument("--selection-fixture", type=Path)
    parser.add_argument("--request-index", type=int, default=1)
    parser.add_argument(
        "--frozen-full-retention",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--turn", type=int, default=4)
    parser.add_argument("--retention-fraction", type=float, default=0.9)
    parser.add_argument("--wire-tail-tokens", type=int, default=32)
    parser.add_argument("--continuation-tokens", type=int, default=16)
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=256,
        help="Chunk size for bounded source and ordinary full-prompt prefill.",
    )
    parser.add_argument("--max-abs-logit-delta", type=float, default=1e-3)
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
