"""Run a frozen seven-turn vLLM scheduler-alias agent-history smoke."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
import vllm
from vllm import LLM, SamplingParams

from pra_hf.agent_executor import (
    configure_append_stable_template,
    validate_append_stable_template,
)
from pra_hf.deployment import PRAWireRequest, PRAWireResource
from pra_vllm.agent_executor import (
    VLLMCudaAgentHistoryExecutor,
    VLLMInProcessSchedulerDriver,
    record_rounded_selected_indices,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_record(module: object) -> dict[str, str]:
    path = Path(inspect.getsourcefile(module) or "").resolve()
    return {"path": str(path), "sha256": _sha256(path)}


def _manifest(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "message_index": index,
            "role": str(row.get("role", "")),
            "content_sha256": hashlib.sha256(
                str(row.get("content", "")).encode("utf-8")
            ).hexdigest(),
        }
        for index, row in enumerate(messages)
    ]


def _request(
    *,
    model: str,
    session_id: str,
    logical: Sequence[Mapping[str, Any]],
    mandatory: Sequence[int],
    selected: Sequence[int],
    retention: float,
    digest: str,
    max_tokens: int,
) -> PRAWireRequest:
    mandatory_set = set(map(int, mandatory))
    resources = tuple(
        PRAWireResource(
            resource_id=f"message-{index}",
            uri=f"pra://agent-history/{session_id}/{index}",
            record_type="agent_history",
            text=str(logical[index].get("content", "")),
            metadata={"message_index": index},
        )
        for index in selected
        if index not in mandatory_set
    )
    return PRAWireRequest(
        model=model,
        messages=tuple(dict(logical[index]) for index in mandatory),
        resources=resources,
        tenant_id="paper4-5-vllm-smoke",
        session_id=session_id,
        max_new_tokens=max_tokens,
        metadata={
            "history_projection": "live-agent-kv-v1",
            "logical_message_manifest": _manifest(logical),
            "mandatory_message_indices": list(map(int, mandatory)),
            "target_retention_fraction": retention,
            "selection_contract": "minimum-retention-floor",
            "segmentation": "causal-assistant-observation-groups",
            "chat_template_digest": digest,
        },
    )


def _run_arm(
    *,
    llm: LLM,
    tokenizer: object,
    driver: VLLMInProcessSchedulerDriver,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    digest: str,
    retention: float,
    turns: int,
    max_tokens: int,
    dense_reference: bool,
) -> dict[str, Any]:
    session_id = f"vllm-agent-bridge-{int(retention * 100)}"
    executor = VLLMCudaAgentHistoryExecutor(
        driver,
        tokenizer,
        model_id=model,
        chat_template_digest=digest,
    )
    logical = [dict(row) for row in messages[:8]]
    observations = [dict(messages[index]) for index in (9, 11, 13, 15, 17, 19)]
    rows: list[dict[str, Any]] = []
    sampling = SamplingParams(temperature=0, max_tokens=max_tokens)
    for turn in range(1, turns + 1):
        if turn > 1:
            logical.append(observations[turn - 2])
        state = executor._sessions.get(session_id)
        if state is None:
            selected = tuple(range(len(logical)))
            selected_pages: tuple[int, ...] | None = None
            mandatory = tuple(range(len(logical)))
        else:
            mandatory = (0, 1, len(logical) - 1)
            if retention >= 1:
                selected = tuple(range(len(logical)))
                selected_pages = tuple(range(state.source_tokens // driver.block_size))
            else:
                selected, selected_pages = record_rounded_selected_indices(
                    logical,
                    state.message_spans,
                    source_tokens=state.source_tokens,
                    block_size=driver.block_size,
                    retention_fraction=retention,
                )
        rendered = tokenizer.apply_chat_template(
            logical, tokenize=True, add_generation_prompt=True
        )
        if isinstance(rendered, Mapping):
            rendered = rendered["input_ids"]
        prompt_ids = rendered.tolist() if hasattr(rendered, "tolist") else list(rendered)
        if prompt_ids and isinstance(prompt_ids[0], list):
            prompt_ids = prompt_ids[0]
        reference = None
        if dense_reference:
            reference = llm.generate(
                {"prompt_token_ids": list(map(int, prompt_ids))},
                sampling,
                use_tqdm=False,
            )[0].outputs[0]
        result = executor.generate(
            _request(
                model=model,
                session_id=session_id,
                logical=logical,
                mandatory=mandatory,
                selected=selected,
                retention=retention,
                digest=digest,
                max_tokens=max_tokens,
            )
        )
        trace = dict(result.trace[0])
        output_tokens = list(map(int, trace.pop("output_token_ids")))
        row = {
            "turn": turn,
            "logical_messages": len(logical),
            "selected_message_indices": list(selected),
            "precomputed_selected_page_indices": (
                None if selected_pages is None else list(selected_pages)
            ),
            "text": result.text,
            "output_token_ids": output_tokens,
            "dense_reference_output_token_ids": (
                None if reference is None else list(map(int, reference.token_ids))
            ),
            "dense_reference_exact": (
                None
                if reference is None
                else output_tokens == list(map(int, reference.token_ids))
            ),
            "trace": trace,
        }
        rows.append(row)
        logical.append({"role": "assistant", "content": result.text})
    before_close = driver.connector._scheduler_alias_registry.snapshot()
    executor.close_session(session_id)
    after_close = driver.connector._scheduler_alias_registry.snapshot()
    return {
        "retention_fraction": retention,
        "rows": rows,
        "lifecycle": {
            "before_close": before_close,
            "after_close": after_close,
            "source_count_after_close": len(after_close.get("sources", {})),
            "active_request_count_after_close": len(after_close.get("active", {})),
            "pending_commit_count_after_close": len(after_close.get("pending_commits", {})),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    trajectory = args.trajectory.expanduser().resolve()
    output = args.output.expanduser().resolve()
    storage = args.storage.expanduser().resolve()
    storage.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(trajectory.read_text(encoding="utf-8"))
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=2,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        enforce_eager=True,
        disable_hybrid_kv_cache_manager=True,
        kv_transfer_config={
            "kv_connector": "PRASparseConnector",
            "kv_connector_module_path": "pra_vllm.cuda_sparse_connector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "storage_path": str(storage),
                "scheduler_page_aliases": True,
            },
        },
    )
    tokenizer = llm.get_tokenizer()
    digest = configure_append_stable_template(tokenizer, "pure-chatml-stable")
    validate_append_stable_template(tokenizer)
    driver = VLLMInProcessSchedulerDriver(llm)
    arms = [
        _run_arm(
            llm=llm,
            tokenizer=tokenizer,
            driver=driver,
            model=args.model,
            messages=frozen["messages"],
            digest=digest,
            retention=fraction,
            turns=args.turns,
            max_tokens=args.max_tokens,
            dense_reference=args.dense_reference,
        )
        for fraction in (1.0, 0.9)
    ]
    rows = [row for arm in arms for row in arm["rows"]]
    sparse_rows = [row for row in arms[1]["rows"] if row["turn"] > 1]
    callbacks = {
        name: sum(int(row["trace"]["scheduler_callback_delta"].get(name, 0)) for row in rows)
        for name in VLLMInProcessSchedulerDriver.REQUIRED_CALLBACKS
    }
    assertions = {
        "requested_turns_completed": all(len(arm["rows"]) == args.turns for arm in arms),
        "pra100_dense_reference_exact": (
            all(row["dense_reference_exact"] for row in arms[0]["rows"])
            if args.dense_reference
            else None
        ),
        "pra90_record_block_floor_valid": all(
            0.9 <= float(row["trace"]["realized_historical_kv_retention"]) < 1.0
            for row in sparse_rows
        ),
        "zero_selected_history_reencoding": all(row["trace"]["selected_history_reencoded_tokens"] == 0 for row in rows),
        "zero_selected_history_copy": all(row["trace"]["selected_history_kv_copy_bytes"] == 0 for row in rows),
        "zero_h2d": all(row["trace"]["host_to_device_bytes"] == 0 for row in rows),
        "zero_total_kv_copy": all(row["trace"]["total_kv_copy_bytes"] == 0 for row in rows),
        "all_alias_callbacks_observed": all(value == 12 for value in callbacks.values()),
        "lifecycle_cleanup": all(
            arm["lifecycle"][key] == 0
            for arm in arms
            for key in (
                "source_count_after_close",
                "active_request_count_after_close",
                "pending_commit_count_after_close",
            )
        ),
    }
    import pra_vllm.agent_executor as agent_module
    import pra_vllm.cuda_scheduler_alias as alias_module
    import pra_vllm.cuda_sparse_connector as connector_module

    payload = {
        "schema_version": "paper4.5.vllm-cuda-stateful-agent-alias-smoke.v1",
        "probe": "frozen_task02_seven_turn_pra100_and_record_block_rounded_pra90",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trajectory": str(trajectory),
        "trajectory_sha256": _sha256(trajectory),
        "model": args.model,
        "engine": "vllm-cuda",
        "engine_version": vllm.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "device": torch.cuda.get_device_name(),
        "temperature": 0,
        "block_size": driver.block_size,
        "turns_per_arm": args.turns,
        "max_new_tokens": args.max_tokens,
        "retention_denominators": {
            "historical_kv": "selected complete resident pages / canonical resident source pages",
            "logical_tokens": "not used for the page-alias correctness gate",
        },
        "copy_accounting": {
            "selected_history_reencoded_tokens": sum(row["trace"]["selected_history_reencoded_tokens"] for row in rows),
            "selected_history_kv_copy_bytes": sum(row["trace"]["selected_history_kv_copy_bytes"] for row in rows),
            "physical_kv_copy_bytes": sum(row["trace"]["physical_kv_copy_bytes"] for row in rows),
            "host_to_device_bytes": sum(row["trace"]["host_to_device_bytes"] for row in rows),
            "total_kv_copy_bytes": sum(row["trace"]["total_kv_copy_bytes"] for row in rows),
            "consumer_temporary_bytes_peak": max(int(row["trace"]["consumer_temporary_bytes"] or 0) for row in rows),
        },
        "consumer": {
            "scheduler_alias_calls": callbacks,
            "fused_sparse_attention_calls": None,
            "fused_sparse_attention_calls_reason": "Scheduler page aliases use vLLM's ordinary attention kernel over aliased block tables; no separate fused PRA kernel is invoked.",
        },
        "arms": arms,
        "assertions": assertions,
        "passed": all(value is not False for value in assertions.values()),
        "scope": {
            "mechanism_only": True,
            "official_swebench_outcome": None,
            "tool_calls": None,
            "first_divergence": None,
            "plain_agent_admission": "not run by this mechanism harness",
        },
        "sources": {
            "agent_executor": _source_record(agent_module),
            "scheduler_alias": _source_record(alias_module),
            "connector": _source_record(connector_module),
        },
    }
    encoded = json.dumps(payload, indent=2) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if not payload["passed"]:
        raise SystemExit(1)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--turns", type=int, default=7)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    parser.add_argument("--dense-reference", action="store_true")
    args = parser.parse_args()
    if args.turns != 7:
        raise ValueError("This frozen qualification requires exactly seven turns.")
    run(args)


if __name__ == "__main__":
    main()
