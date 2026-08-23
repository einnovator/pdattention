"""Confirm selected request/reply policies through native K/V and frozen decode.

The retrieval study covers the full 74-example test cohort.  This deliberately
smaller, dataset-stratified confirmation maps the selected chunk identities to
the model's native layer K/V, measures physical materialization, teacher-forced
answer likelihood, and deterministic generation quality.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_6_hybrid_pra.run_channel_geometry import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    _load_cases,
)
from experiments.paper3_5_adaptive_pra.request_reply_full_surface import (  # noqa: E402
    _resolve_local,
)
from experiments.paper3_kv_materialization.run_oracle_frontier import (  # noqa: E402
    _generate,
    _prompt,
    _row_metrics,
    _teacher_forced,
)
from pra_hf import PRAConfig, PRAForCausalLM  # noqa: E402
from pra_hf.output_validation import deterministic_answer_metrics  # noqa: E402
from pra_torch.memory import SelectedChunk  # noqa: E402


DATASETS = ("hotpotqa", "qasper", "2wikimultihopqa", "musique")
POLICIES = (
    "B0_validation_fixed",
    "B1_one_shot_no_graph",
    "B3_callback_no_graph",
    "B5t_threshold_graph",
    "B6_complete_action_oracle",
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write(path: Path, rows) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _checkpoint(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _controller_rows(path: Path) -> dict[tuple[str, str, str], dict]:
    values = _read(path)
    selected = {}
    for row in values:
        if row["protocol"] != "standard" or int(row["router_seed"]) != 42:
            continue
        if row["baseline"] not in POLICIES:
            continue
        if row["baseline"] == "B3_callback_no_graph" and row.get("feature_ablation") != "compact_state":
            continue
        selected[(row["dataset"], row["example_id"], row["baseline"])] = row
    return selected


def _cases(args, selected_rows):
    loader_args = SimpleNamespace(
        cache_dir=args.cache_dir,
        seed=args.seed,
        paper2_feature_dir=args.paper2_feature_dir,
        natural_features=args.natural_features,
        musique_dev=args.musique_dev,
        twowiki_dev=args.twowiki_dev,
    )
    candidates = [
        (feature, example)
        for feature, example in _load_cases(loader_args)
        if feature["split"] == "test"
        and (feature["dataset"], feature["example_id"], "B0_validation_fixed") in selected_rows
    ]
    output = []
    for dataset in DATASETS:
        local = sorted(
            (row for row in candidates if row[0]["dataset"] == dataset),
            key=lambda row: row[0]["example_id"],
        )[: args.examples_per_dataset]
        if len(local) != args.examples_per_dataset:
            raise ValueError(f"Expected {args.examples_per_dataset} {dataset} examples, found {len(local)}.")
        output.extend(local)
    return output


def _selected(
    entry,
    routing_layer: int,
    identities: list[str],
    conceptual_spans,
) -> list[SelectedChunk]:
    chunks = {chunk.chunk_id: chunk for chunk in entry.layer_memory[routing_layer].chunks}
    resolved = []
    for identity in identities:
        if identity in chunks:
            matches = [chunks[identity]]
            conceptual_span = (matches[0].logical_start, matches[0].logical_end)
        else:
            try:
                conceptual_index = int(identity.rsplit("#chunk=", 1)[1])
                conceptual_span = tuple(map(int, conceptual_spans[conceptual_index]))
            except (IndexError, TypeError, ValueError) as error:
                raise KeyError(f"Cannot resolve conceptual chunk identity {identity!r}.") from error
            start, end = conceptual_span
            matches = [
                chunk
                for chunk in chunks.values()
                if chunk.logical_start < end and start < chunk.logical_end
            ]
        if not matches:
            raise KeyError(
                f"Conceptual span {conceptual_span} has no native K/V chunk mapping."
            )
        for chunk in matches:
            if all(chunk.chunk_id != prior[0].chunk_id for prior in resolved):
                resolved.append((chunk, identity, conceptual_span))
    return [
        SelectedChunk(
            entry=entry,
            chunk=chunk,
            reference_score=1.0,
            chunk_score=1.0 - rank * 1e-6,
            layer_id=routing_layer,
            reference_rank=1,
            rank_within_reference=rank,
            metadata={
                "selection_source": "measured_request_reply_surface",
                "conceptual_chunk_id": conceptual_identity,
                "conceptual_span": conceptual_span,
            },
        )
        for rank, (chunk, conceptual_identity, conceptual_span) in enumerate(resolved, 1)
    ]


def _aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["condition"])].append(row)
    metrics = (
        "evidence_recall",
        "precision",
        "exact_match",
        "token_f1",
        "answer_contained",
        "gold_mean_token_logprob",
        "gold_logprob_delta_vs_no_memory",
        "materialized_unique_tokens",
        "native_kv_token_states",
        "native_kv_bytes",
        "active_kv_fraction",
        "teacher_forced_seconds",
        "total_generation_seconds",
        "tokens_per_second",
    )
    return [
        {
            "dataset": key[0],
            "condition": key[1],
            "examples": len(values),
            **{
                metric: statistics.fmean(float(row[metric]) for row in values)
                for metric in metrics
            },
        }
        for key, values in sorted(grouped.items())
    ]


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    selected_rows = _controller_rows(args.controller_rows)
    cases = _cases(args, selected_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "end_to_end_checkpoint.jsonl"
    rows = _checkpoint(checkpoint)
    for row in rows:
        row.setdefault("conceptual_selected_chunks", 0)
        row.setdefault("physical_selected_chunks", 0)
    complete = {(row["dataset"], row["example_id"], row["condition"]) for row in rows}

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model_cache = (
        {"cache_dir": args.model_cache_dir}
        if args.model_cache_dir is not None
        else {}
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=args.local_files_only,
        **model_cache,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=args.local_files_only,
        attn_implementation="eager",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        **model_cache,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = tuple(range(model.config.num_hidden_layers - args.consumption_layers, model.config.num_hidden_layers))
    pra = PRAForCausalLM.from_model(
        model,
        tokenizer,
        pra_config=PRAConfig(
            routing_layer=model.config.num_hidden_layers - 1,
            consumption_layers=layers,
            chunk_tokens=args.chunk_tokens,
            selected_fraction=None,
            top_k=1,
            max_direct_context=args.prompt_tokens,
            native_operation_limit=args.native_limit,
            max_materialized_tokens=args.materialized_tokens,
            context_safety_reserve_tokens=0,
            encoding_block_tokens=args.encoding_tokens,
            reference_device="cpu",
            pin_reference_memory=device.type == "cuda",
            non_blocking_transfer=device.type == "cuda",
        ),
    )

    conditions = ("no_memory", *POLICIES)
    for case_index, (feature, example) in enumerate(cases, 1):
        dataset, example_id = feature["dataset"], feature["example_id"]
        pending = [name for name in conditions if (dataset, example_id, name) not in complete]
        if not pending:
            continue
        pra.clear_references()
        uri = f"benchmark://{dataset}/{example_id}"
        pra.add_reference(example["source"], uri=uri)
        entry = pra._handle.cache.get(uri)
        if entry is None:
            raise AssertionError("Reference cache entry missing after native K/V encoding.")
        encoded = _prompt(tokenizer, example["question"])
        for condition in pending:
            retrieval = selected_rows.get((dataset, example_id, condition))
            identities: list[str] = []
            hits: list[SelectedChunk] = []
            if condition == "no_memory":
                active_layers = ()
                pra._handle.configure_memory_layers(set())
                positions = []
            else:
                identities = [value for value in retrieval["selected_chunk_ids"].split("|") if value]
                hits = _selected(
                    entry,
                    pra.routing_layer,
                    identities,
                    feature["chunk_spans"],
                )
                mapped = pra._handle.map_chunk_identities_to_layers([hits], layers)
                pra._handle.pra_config.detail_materialization = "selected_chunks"
                pra._handle.configure_memory_layers(set(layers), fixed_selections=mapped)
                active_layers = layers
                positions = [position for hit in hits for position in range(hit.logical_start, hit.logical_end)]
            teacher = _teacher_forced(
                pra,
                tokenizer,
                encoded,
                str(example["answer"]),
                device,
                positions,
                [tuple(map(int, span)) for span in feature["evidence_spans"]],
                active_layers,
            )
            generation = _generate(pra.model, tokenizer, encoded, device, args.max_new_tokens)
            physical = _row_metrics(teacher.pop("diagnostics_by_layer"), active_layers)
            row = {
                "dataset": dataset,
                "example_id": example_id,
                "condition": condition,
                "reference_answer": example["answer"],
                "evidence_recall": float(retrieval["evidence_recall"]) if retrieval else 0.0,
                "precision": float(retrieval["precision"]) if retrieval else 0.0,
                "selected_chunk_ids": retrieval["selected_chunk_ids"] if retrieval else "",
                "conceptual_selected_chunks": len(identities) if retrieval else 0,
                "physical_selected_chunks": len(hits) if retrieval else 0,
                "materialization_layers": "|".join(map(str, active_layers)),
                "logical_source_tokens": int(feature["source_tokens"]),
                "active_kv_fraction": physical["materialized_unique_tokens"] / max(int(feature["source_tokens"]), 1),
                **physical,
                **teacher,
                **generation,
                **deterministic_answer_metrics(generation["generated_answer"], str(example["answer"])),
            }
            _append(checkpoint, row)
            rows.append(row)
            complete.add((dataset, example_id, condition))
        pra.clear_references()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[end-to-end {case_index}/{len(cases)}] {dataset} {example_id}", flush=True)

    baselines = {(row["dataset"], row["example_id"]): row for row in rows if row["condition"] == "no_memory"}
    for row in rows:
        row["gold_logprob_delta_vs_no_memory"] = row["gold_mean_token_logprob"] - baselines[(row["dataset"], row["example_id"])]["gold_mean_token_logprob"]
    aggregates = _aggregate(rows)
    _write(args.output / "end_to_end_rows.csv", rows)
    _write(args.output / "end_to_end_summary.csv", aggregates)
    artifact = {
        "schema_version": "1.0",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "backbone_frozen": True,
        "retrieval_test_examples": 74,
        "end_to_end_examples": len(cases),
        "examples_per_dataset": args.examples_per_dataset,
        "conditions": list(conditions),
        "routing_layer": pra.routing_layer,
        "consumption_layers": list(layers),
        "chunk_tokens": args.chunk_tokens,
        "max_materialized_tokens": args.materialized_tokens,
        "rows": len(rows),
        "aggregates": aggregates,
    }
    (args.output / "end_to_end_manifest.json").write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    inherited = ROOT.parent / "pdattention-iter-gist"
    primary = ROOT.parent / "pdattention"
    paper2 = primary / "docs/papers/shared/results/paper2_6_hybrid_pra/channel_selection"
    root_callback = ROOT / "docs/papers/shared/results/paper3_5_adaptive_pra/root_callback"
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--examples-per-dataset", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--chunk-tokens", type=int, default=32)
    parser.add_argument("--encoding-tokens", type=int, default=256)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--native-limit", type=int, default=4096)
    parser.add_argument("--materialized-tokens", type=int, default=128)
    parser.add_argument("--consumption-layers", type=int, default=8)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/.hf_cache")
    parser.add_argument("--model-cache-dir", type=Path)
    parser.add_argument("--controller-rows", type=Path, default=root_callback / "controller/controller_rows.csv")
    parser.add_argument("--paper2-feature-dir", type=Path, default=_resolve_local(ROOT / "docs/papers/shared/results/paper2_hf/routing/learned_adapter", "pdattention", "pdattention-iter-gist"))
    parser.add_argument("--natural-features", type=Path, default=_resolve_local(ROOT / "docs/papers/shared/results/paper2_5_iterative_pra/natural_graph_depth/natural_graph_features.pt", "pdattention-iter-gist"))
    parser.add_argument("--musique-dev", type=Path, default=inherited / "data/.paper2_5_datasets/musique/data/musique_ans_v1.0_dev.jsonl")
    parser.add_argument("--twowiki-dev", type=Path, default=inherited / "data/.paper2_5_datasets/2wiki/dev.json")
    parser.add_argument("--output", type=Path, default=root_callback / "end_to_end")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
