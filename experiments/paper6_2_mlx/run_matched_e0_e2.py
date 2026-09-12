"""Compare matched selected-text and native-K/V execution on MLX-LM."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

from experiments.engine_serving.matched_e0_e2_contract import (
    SCHEMA_VERSION,
    benchmark_metrics,
    benchmark_row,
    regime_schedule,
    validate_payload,
)
from experiments.engine_serving.matched_qa import load_matched_examples
from experiments.paper6_2_mlx.run_answer_quality_pressure import (
    _answer_logprob,
    _bounded_source,
    _metrics,
)


# Keep the MLX runner independent of importing the top-level ``pra_hf``
# package, whose initialization includes PyTorch-only modules.
CURRENT_ENGINE_EVIDENCE_CONTRACT = "live-kv-original-position-lifecycle-v1"


def _git_state() -> tuple[str | None, bool | None, str | None]:
    """Return the exact source revision and whether tracked files differ."""

    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), capture_output=True, text=True, check=False
    )
    if commit.returncode:
        return None, None, None
    diff = subprocess.run(
        ("git", "diff", "--binary", "HEAD", "--"),
        capture_output=True,
        check=False,
    )
    if diff.returncode:
        return commit.stdout.strip(), None, None
    dirty = bool(diff.stdout)
    return (
        commit.stdout.strip(),
        dirty,
        hashlib.sha256(diff.stdout).hexdigest() if dirty else None,
    )


def _cache_snapshot(cache):
    """Trim an ordinary prefix cache to immutable state for request forks."""

    import mlx.core as mx

    states = []
    for layer in cache:
        state = layer.state
        states.append(tuple(mx.array(value) for value in state))
    mx.eval(states)
    return tuple(states)


def _restore_cache(model, states):
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    if len(cache) != len(states):
        raise RuntimeError("Ordinary MLX prefix cache changed layer count.")
    for layer, state in zip(cache, states):
        layer.state = state
    return cache


def _cache_nbytes(states) -> int:
    return sum(value.nbytes for state in states for value in state[:2])


def _last_logits(model, token_ids, cache):
    import mlx.core as mx

    logits = model(mx.array(token_ids, dtype=mx.int32)[None], cache=cache)
    result = logits[0, -1].astype(mx.float32)
    mx.eval(result)
    return result


def _owned_native_copy(memory):
    """Build an owned value-identical oracle for the same MLX consumer."""

    import mlx.core as mx
    from pra_mlx.native import MLXNativeLayerKV, MLXNativeMemory

    layers = []
    for layer in memory.layers:
        zero_k = mx.array(0, dtype=layer.keys.dtype)
        zero_v = mx.array(0, dtype=layer.values.dtype)
        layers.append(
            MLXNativeLayerKV(layer.keys + zero_k, layer.values + zero_v)
        )
    result = MLXNativeMemory(tuple(layers), memory.source_tokens)
    mx.eval(*(value for layer in result.layers for value in (layer.keys, layer.values)))
    return result


def _generate_timed(model, tokenizer, query, cache, max_tokens: int):
    """Return deterministic generation with TTFT and mean inter-token latency."""

    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    started = time.perf_counter()
    generated = []
    arrivals = []
    stop_ids = _stop_token_ids(tokenizer)
    for token, _ in generate_step(
        mx.array(query, dtype=mx.int32),
        model,
        max_tokens=max_tokens,
        prompt_cache=cache,
        sampler=make_sampler(temp=0),
    ):
        token = int(token)
        if token in stop_ids:
            break
        generated.append(token)
        arrivals.append((time.perf_counter() - started) * 1000.0)
    completion_ms = (time.perf_counter() - started) * 1000.0
    itl_ms = (
        sum(right - left for left, right in zip(arrivals, arrivals[1:]))
        / (len(arrivals) - 1)
        if len(arrivals) > 1
        else 0.0
    )
    return {
        "output": tokenizer.decode(generated).strip(),
        "output_token_ids": generated,
        "generated_tokens": len(generated),
        "ttft_ms": arrivals[0] if arrivals else completion_ms,
        "itl_ms": itl_ms,
        "completion_latency_ms": completion_ms,
    }


def _stop_token_ids(tokenizer) -> set[int]:
    """Normalize single- and multi-EOS tokenizer contracts used by MLX-LM."""

    values = getattr(tokenizer, "eos_token_ids", None)
    if values is None:
        values = getattr(tokenizer, "eos_token_id", None)
    if values is None:
        return set()
    if isinstance(values, int):
        return {values}
    return {int(value) for value in values}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=("qasper", "hotpotqa", "2wikimultihopqa"), required=True
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/papers/shared/results/matched_e0_e2_qa_manifest.json"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "pra")
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-source-tokens", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument(
        "--reuse-max-new-tokens",
        type=int,
        default=0,
        help="Optional shorter decode for non-cold reuse probes.",
    )
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--warm-repeats", type=int, default=2)
    parser.add_argument("--multi-query-count", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--hardware-label", default="NOT_RECORDED")
    parser.add_argument(
        "--timing-status",
        choices=("PREQUALIFICATION", "QUALIFIED"),
        default="PREQUALIFICATION",
        help="Timings are never qualified implicitly from a functional run.",
    )
    parser.add_argument(
        "--materialization",
        choices=("segmented", "concatenated"),
        default="segmented",
        help="Segmented is the corrected no-K/V-concatenation consumer path.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import mlx.core as mx
    import mlx_lm
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from pra_hf.live_history import LiveKVSelectionPlan
    from pra_mlx.native import capture_live_native_memory, make_native_prompt_cache
    from pra_mlx.qwen3_segmented import install_qwen3_segmented_attention

    manifest, examples = load_matched_examples(
        args.manifest, args.dataset, args.cache_dir
    )
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    model, tokenizer = load(args.model, revision=args.revision)
    segmented = args.materialization == "segmented"
    if segmented:
        install_qwen3_segmented_attention(model)
    model_runner_lock = threading.RLock()
    pra_commit, source_tree_dirty, source_tree_diff_sha256 = _git_state()
    rows = []
    same_consumer_gates = []
    for example in examples:
        prepared_at = time.perf_counter()
        candidate_tokens = list(
            tokenizer.encode(example.candidate_source, add_special_tokens=False)
        )
        source = _bounded_source(tokenizer, example.selected_source, args.max_source_tokens)
        text_preparation_ms = (time.perf_counter() - prepared_at) * 1000.0
        answer = list(tokenizer.encode(" " + example.answer, add_special_tokens=False))

        started = time.perf_counter()
        ordinary_cache = make_prompt_cache(model)
        model(mx.array(source, dtype=mx.int32)[None], cache=ordinary_cache)
        mx.eval([layer.state for layer in ordinary_cache])
        source_encode_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        resident_selection = capture_live_native_memory(
            ordinary_cache, LiveKVSelectionPlan.full(len(source))
        )
        native_memory = resident_selection.memory
        selection_capture_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        ordinary_states = _cache_snapshot(ordinary_cache)
        prefix_snapshot_ms = (time.perf_counter() - started) * 1000.0

        requests = regime_schedule(
            example.question,
            warm_repeats=args.warm_repeats,
            multi_query_count=args.multi_query_count,
            concurrency=args.concurrency,
        )
        regular_requests = tuple(
            request
            for request in requests
            if request.regime != "concurrent_shared_resource"
        )
        concurrent_requests = tuple(
            request
            for request in requests
            if request.regime == "concurrent_shared_resource"
        )

        # Correctness is selection/materialization parity, not equality across
        # two different attention implementations. Compare the live prefix
        # alias with an owned, value-identical K/V oracle while holding the
        # segmented consumer and original positions fixed.
        gate_query = list(
            tokenizer.encode(regular_requests[0].query.text, add_special_tokens=False)
        )
        owned_reference = _owned_native_copy(native_memory)
        with model_runner_lock:
            candidate_logits = _last_logits(
                model,
                gate_query,
                make_native_prompt_cache(model, native_memory, segmented=segmented),
            )
            reference_logits = _last_logits(
                model,
                gate_query,
                make_native_prompt_cache(model, owned_reference, segmented=segmented),
            )
            same_consumer_delta = float(
                mx.max(mx.abs(candidate_logits - reference_logits)).item()
            )
        same_consumer_gates.append(
            {
                "dataset": example.dataset,
                "example_id": example.example_id,
                "reference": (
                    "owned value-identical K/V; same segmented attention consumer"
                ),
                "max_abs_logit_delta": same_consumer_delta,
                "passed": same_consumer_delta == 0.0,
            }
        )
        del owned_reference

        def execute(request, condition):
            query = list(
                tokenizer.encode(request.query.text, add_special_tokens=False)
            )
            cache_factory = (
                (lambda: _restore_cache(model, ordinary_states))
                if condition == "e0_selected_text"
                else (
                    lambda: make_native_prompt_cache(
                        model, native_memory, segmented=segmented
                    )
                )
            )
            # mlx-lm's in-process model is not re-entrant. Multiple requests
            # may coexist, but execution on one model is deliberately
            # serialized and reported as such rather than mislabeled as a
            # concurrent serving-engine result.
            with model_runner_lock:
                active_before = int(mx.get_active_memory())
                reset_peak = getattr(mx, "reset_peak_memory", None)
                if reset_peak is not None:
                    reset_peak()
                logprob = _answer_logprob(model, query, answer, cache_factory())
                generation_tokens = (
                    args.max_new_tokens
                    if request.regime == "cold_one_shot"
                    or args.reuse_max_new_tokens <= 0
                    else args.reuse_max_new_tokens
                )
                generated = _generate_timed(
                    model, tokenizer, query, cache_factory(), generation_tokens
                )
                active_after = int(mx.get_active_memory())
                peak = int(mx.get_peak_memory())
            exact, f1 = _metrics(generated["output"], example.answer)
            memory = {
                "consumer_active_before_bytes": active_before,
                "consumer_active_after_bytes": active_after,
                "consumer_peak_bytes": peak,
                "consumer_temporary_peak_delta_bytes": max(0, peak - active_before),
            }
            return request, condition, query, generated, logprob, exact, f1, memory

        def append_result(result, requests_per_second=None):
            request, condition, query, generated, logprob, exact, f1, memory = result
            native = condition == "e2_native_kv"
            reused = request.regime != "cold_one_shot"
            rows.append(
                benchmark_row(
                    condition=condition,
                    selection=example.selection,
                    request=request,
                    output=str(generated["output"]),
                    metrics=benchmark_metrics(
                        exact_match=exact,
                        token_f1=f1,
                        gold_answer_logprob=logprob,
                        evidence_recall=example.evidence_recall,
                        candidate_tokens=len(candidate_tokens),
                        selected_source_tokens=len(source),
                        visible_prompt_tokens=(
                            len(query) if native else len(source) + len(query)
                        ),
                        selected_native_kv_tokens=len(source) if native else 0,
                        active_detail_bytes=native_memory.nbytes if native else 0,
                        retained_detail_bytes=native_memory.nbytes if native else 0,
                        text_preparation_ms=text_preparation_ms,
                        kv_encode_ms=source_encode_ms,
                        index_construction_ms=0.0,
                        time_to_usable_context_ms=(
                            text_preparation_ms
                            + source_encode_ms
                            + (selection_capture_ms if native else prefix_snapshot_ms)
                        ),
                        ttft_ms=float(generated["ttft_ms"]),
                        itl_ms=float(generated["itl_ms"]),
                        total_latency_ms=float(generated["completion_latency_ms"]),
                        generated_tokens=int(generated["generated_tokens"]),
                        ordinary_prefix_cache_hit_tokens=(
                            len(source) if not native else 0
                        ),
                        pra_hot_hit=native and reused,
                        pra_warm_hit=False,
                        bytes_read=native_memory.nbytes if native else 0,
                        bytes_promoted=0,
                        # The legacy schema requires numeric fields. Zero is
                        # conservative: the corrected runner does not infer
                        # physical savings from logical reuse.
                        bytes_avoided=0,
                        duplicate_physical_kv_avoided_bytes=0,
                        requests_per_second=requests_per_second,
                    ),
                    extra={
                        "dataset": example.dataset,
                        "seed": example.seed,
                        "gold_answer": example.answer,
                        "execution_source_sha256": example.selected_source_sha256,
                        "e0_prefix_kv_bytes": _cache_nbytes(ordinary_states),
                        "source_kv_captured_once_from_live_prefix": True,
                        "source_encode_ms": source_encode_ms,
                        "selection_capture_ms": selection_capture_ms,
                        "ordinary_prefix_snapshot_ms": prefix_snapshot_ms,
                        "selection_physical_kv_copy": (
                            resident_selection.physical_kv_copy
                        ),
                        "selected_history_reencoded_tokens": 0,
                        "selection_materialization_kv_copy_bytes": 0 if native and segmented else None,
                        "physical_kv_copy_bytes": None,
                        "physical_copy_accounting": "NOT_MEASURED",
                        "consumer_memory": memory,
                        "concurrency_execution": (
                            "concurrent_borrows_serialized_model_runner"
                            if request.regime == "concurrent_shared_resource"
                            else "single_request"
                        ),
                    },
                )
            )

        for request in regular_requests:
            for condition in ("e0_selected_text", "e2_native_kv"):
                append_result(execute(request, condition))
        for condition in ("e0_selected_text", "e2_native_kv"):
            started = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.concurrency
            ) as pool:
                results = tuple(
                    pool.map(
                        lambda request: execute(request, condition),
                        concurrent_requests,
                    )
                )
            wall_ms = (time.perf_counter() - started) * 1000.0
            throughput = len(results) / max(wall_ms / 1000.0, 1e-9)
            for result in results:
                append_result(result, throughput)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "paper6_cross_engine_matched_e0_e2_mlx_v2",
        "evidence_tier": "NATURAL_QA_MATCHED_SELECTION",
        "engine": "mlx-lm",
        "engine_version": getattr(mlx_lm, "__version__", "unknown"),
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "hardware": args.hardware_label,
        "timing_status": args.timing_status,
        "model_id": args.model,
        "model_revision": args.revision,
        "dataset": args.dataset,
        "cohort": manifest["cohort"],
        "selection_policy": manifest["selection_policy"],
        "max_source_tokens": args.max_source_tokens,
        "warm_repeats": args.warm_repeats,
        "multi_query_count": args.multi_query_count,
        "concurrency": args.concurrency,
        "concurrency_semantics": "concurrent_borrows_serialized_model_runner",
        "materialization": args.materialization,
        "selected_history_reencoded_tokens": 0,
        "physical_kv_copy_bytes": None,
        "physical_copy_accounting": "NOT_MEASURED",
        "pra_commit": pra_commit,
        "source_tree_dirty": source_tree_dirty,
        "source_tree_diff_sha256": source_tree_diff_sha256,
        "engine_gate_commit": pra_commit,
        "engine_contract_version": (
            CURRENT_ENGINE_EVIDENCE_CONTRACT
            if segmented
            else "legacy-concatenated-kv-v1"
        ),
        "same_consumer_correctness_gates": same_consumer_gates,
        "same_consumer_correctness_passed": all(
            row["passed"] for row in same_consumer_gates
        ),
        "rows": rows,
    }
    validate_payload(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
