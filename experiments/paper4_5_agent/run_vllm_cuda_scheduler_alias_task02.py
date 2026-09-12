"""Qualify scheduler-owned zero-copy CUDA aliases on frozen Task02 turns."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import time
import uuid
from pathlib import Path

os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
import vllm
from vllm import LLM, SamplingParams

from experiments.paper4_5_agent.run_hf_agent_cache_equivalence import _assistant_prompts
from experiments.paper4_5_agent.run_vllm_cuda_live_agent_kv_gate import (
    _selected_page_indices,
)
from experiments.paper4_5_agent.sparse_gate_common import sparse_causal_plan
from experiments.paper6_vllm.run_cuda_connector_candidate import _prompt
from pra_vllm.cuda_sparse_protocol import SparseCudaConnectorCommand


def _source_record(module: object) -> dict[str, str]:
    path = Path(inspect.getsourcefile(module) or "").resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _timed_generate(
    llm: LLM,
    prompts: object,
    sampling: SamplingParams,
) -> tuple[list[object], float]:
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    torch.cuda.synchronize()
    return outputs, (time.perf_counter_ns() - started) / 1_000_000


def run(args: argparse.Namespace) -> dict[str, object]:
    trajectory_path = args.trajectory.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    storage = args.storage.expanduser().resolve()
    storage.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]

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
    block_size = int(llm.llm_engine.vllm_config.cache_config.block_size)
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    messages = trajectory["messages"]
    prompts = _assistant_prompts(tokenizer, trajectory, args.turns)
    assistant_indexes = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "assistant"
    ][: args.turns]
    prime_sampling = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
    continuation_sampling = SamplingParams(
        temperature=0,
        max_tokens=args.continuation_tokens,
        ignore_eos=True,
    )
    scheduler_connector = llm.llm_engine.engine_core.engine_core.scheduler.connector
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []

    for turn, (assistant_index, prompt) in enumerate(
        zip(assistant_indexes, prompts), start=1
    ):
        source_tokens = (len(prompt) // block_size) * block_size
        if source_tokens == 0 or source_tokens == len(prompt):
            skipped.append({"turn": turn, "reason": "no_nonempty_page_aligned_suffix"})
            continue
        source, suffix = prompt[:source_tokens], prompt[source_tokens:]
        try:
            plan = sparse_causal_plan(
                tokenizer,
                messages[:assistant_index],
                prompt,
                source_tokens=source_tokens,
                retention_fraction=args.retention_fraction,
            )
        except RuntimeError as error:
            skipped.append({"turn": turn, "reason": str(error)})
            continue
        page_indices = _selected_page_indices(plan, block_size)
        selected_tokens = len(page_indices) * block_size
        if not page_indices:
            skipped.append({"turn": turn, "reason": "selection_has_no_complete_pages"})
            continue
        if args.retention_fraction < 1 and selected_tokens == source_tokens:
            skipped.append({"turn": turn, "reason": "selection_has_no_page_holes"})
            continue

        selected = [
            token
            for page_index in page_indices
            for token in source[
                page_index * block_size : (page_index + 1) * block_size
            ]
        ]
        source_key = f"task02-{run_id}-turn{turn}-full"
        selected_key = f"task02-{run_id}-turn{turn}-selected"
        generation = turn
        prime_command = SparseCudaConnectorCommand(
            "store",
            source_key,
            source_tokens,
            source_tokens,
            source_generation=generation,
            residency="warm",
            request_scope=f"task02-turn{turn}-prime",
        )
        _, prime_ms = _timed_generate(
            llm,
            _prompt(source + suffix[:1], prime_command.cache_salt()),
            prime_sampling,
        )
        source_snapshot = scheduler_connector._scheduler_alias_registry.snapshot()
        source_row = source_snapshot["sources"].get(source_key)
        if source_row is None:
            raise RuntimeError(f"CUDA source {source_key!r} was not pinned.")

        directory = storage / hashlib.sha256(selected_key.encode()).hexdigest()
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "pra-vllm-cuda-scheduler-alias-v1",
                    "logical_key": selected_key,
                    "source_tokens": selected_tokens,
                    "source_generation": generation,
                    "parent_logical_key": source_key,
                    "selected_page_indices": list(page_indices),
                    "physical_kv_copy_bytes": 0,
                    "host_to_device_bytes": 0,
                }
            ),
            encoding="utf-8",
        )
        load_command = SparseCudaConnectorCommand(
            "load",
            selected_key,
            selected_tokens,
            source_tokens,
            source_generation=generation,
            residency="hot",
            request_scope=f"task02-turn{turn}-pair",
        )
        before_pair = scheduler_connector._scheduler_alias_registry.snapshot()
        outputs, pair_ms = _timed_generate(
            llm,
            [
                _prompt(selected + suffix, load_command.cache_salt()),
                _prompt(selected + suffix, load_command.cache_salt()),
            ],
            continuation_sampling,
        )
        after_pair = scheduler_connector._scheduler_alias_registry.snapshot()
        token_rows = [list(map(int, row.outputs[0].token_ids)) for row in outputs]
        evicted = scheduler_connector.evict_scheduler_source(
            source_key, source_generation=generation
        )
        after_evict = scheduler_connector._scheduler_alias_registry.snapshot()
        before_telemetry = before_pair["telemetry"]
        after_telemetry = after_pair["telemetry"]
        telemetry_delta = {
            key: int(after_telemetry[key]) - int(before_telemetry[key])
            for key in after_telemetry
        }
        has_holes = any(
            right != left + 1
            for left, right in zip(page_indices, page_indices[1:])
        ) or selected_tokens < source_tokens
        rows.append(
            {
                "turn": turn,
                "assistant_message_index": assistant_index,
                "requested_retention_fraction": args.retention_fraction,
                "realized_retention_fraction": selected_tokens / source_tokens,
                "block_size": block_size,
                "source_tokens": source_tokens,
                "selected_tokens": selected_tokens,
                "source_position_base": source_tokens,
                "wire_query_suffix_tokens": len(suffix),
                "source_block_count": source_tokens // block_size,
                "selected_block_count": len(page_indices),
                "selected_page_indices": list(page_indices),
                "has_holes": has_holes,
                "selection_plan": plan.to_dict(),
                "source_block_ids": source_row["block_ids"],
                "same_subset_output_token_ids": token_rows,
                "same_subset_exact": token_rows[0] == token_rows[1],
                "prime_ms": prime_ms,
                "concurrent_pair_ms": pair_ms,
                "telemetry_delta": telemetry_delta,
                "evicted_source_block_ids": list(evicted),
                "source_removed_after_evict": source_key not in after_evict["sources"],
                "copy_accounting": {
                    "physical_kv_copy_bytes": 0,
                    "host_to_device_bytes": 0,
                    "selected_history_reencoded_tokens": 0,
                },
            }
        )

    final_snapshot = scheduler_connector._scheduler_alias_registry.snapshot()

    import pra_vllm.cuda_scheduler_alias as alias_module
    import pra_vllm.cuda_sparse_connector as connector_module
    import vllm.v1.core.sched.scheduler as scheduler_module

    payload: dict[str, object] = {
        "schema_version": "paper4.5.vllm-cuda-scheduler-page-alias-task02.v2",
        "probe": "frozen_task02_full_requested_090_scheduler_owned_page_alias",
        "experiment_revision": os.environ.get(
            "PRA_EXPERIMENT_REVISION", "uncommitted"
        ),
        "trajectory": str(trajectory_path),
        "trajectory_sha256": hashlib.sha256(trajectory_path.read_bytes()).hexdigest(),
        "engine": "vllm-cuda",
        "engine_version": vllm.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "device": torch.cuda.get_device_name(),
        "model": args.model,
        "temperature": 0,
        "requested_turns": args.turns,
        "expected_eligible_turns": args.expected_eligible_turns,
        "completed_turns": len(rows),
        "exact_turns": sum(int(row["same_subset_exact"]) for row in rows),
        "requested_retention_fraction": args.retention_fraction,
        "realized_retention_fraction_min": min(
            (float(row["realized_retention_fraction"]) for row in rows),
            default=None,
        ),
        "realized_retention_fraction_max": max(
            (float(row["realized_retention_fraction"]) for row in rows),
            default=None,
        ),
        "copy_accounting": {
            "physical_kv_copy_bytes": 0,
            "host_to_device_bytes": 0,
            "selected_history_reencoded_tokens": 0,
        },
        "rows": rows,
        "skipped_turns": skipped,
        "final_registry_snapshot": final_snapshot,
        "integration_scope": {
            "scheduler_block_table_authoritative": True,
            "original_position_persisted_on_every_decode_step": True,
            "complete_pages_required": True,
            "homogeneous_single_kv_group_required": True,
            "in_process_v1_scheduler_required": True,
            "multiprocess_scheduler_worker_qualification": False,
        },
        "provenance": {
            "experiment": {
                "path": str(Path(__file__).resolve()),
                "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            "alias_module": _source_record(alias_module),
            "connector_module": _source_record(connector_module),
            "pinned_scheduler": _source_record(scheduler_module),
        },
    }
    telemetry = final_snapshot["telemetry"]
    payload["qualified"] = bool(
        len(rows) == args.expected_eligible_turns
        and all(row["same_subset_exact"] for row in rows)
        and all(row["has_holes"] for row in rows)
        and all(row["telemetry_delta"]["alias_install_events"] == 2 for row in rows)
        and all(row["telemetry_delta"]["alias_release_events"] == 2 for row in rows)
        and all(
            len(row["evicted_source_block_ids"]) == row["source_block_count"]
            for row in rows
        )
        and all(row["source_removed_after_evict"] for row in rows)
        and int(telemetry["source_pin_events"]) == len(rows)
        and int(telemetry["alias_install_events"]) == 2 * len(rows)
        and int(telemetry["alias_release_events"]) == 2 * len(rows)
        and int(telemetry["physical_kv_copy_bytes"]) == 0
        and int(telemetry["host_to_device_bytes"]) == 0
        and int(telemetry["selected_history_reencoded_tokens"]) == 0
        and not final_snapshot["sources"]
        and not final_snapshot["active_requests"]
        and not final_snapshot["pending_requests"]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--storage", type=Path, default=Path(".pra/vllm-cuda-task02-alias")
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--expected-eligible-turns", type=int, default=7)
    parser.add_argument("--continuation-tokens", type=int, default=16)
    parser.add_argument("--retention-fraction", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    args = parser.parse_args()
    payload = run(args)
    print(
        json.dumps(
            {
                "qualified": payload["qualified"],
                "completed_turns": payload["completed_turns"],
                "exact_turns": payload["exact_turns"],
                "realized_retention_fraction_min": payload[
                    "realized_retention_fraction_min"
                ],
                "realized_retention_fraction_max": payload[
                    "realized_retention_fraction_max"
                ],
                "final_telemetry": payload["final_registry_snapshot"]["telemetry"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    raise SystemExit(0 if payload["qualified"] else 1)


if __name__ == "__main__":
    main()
