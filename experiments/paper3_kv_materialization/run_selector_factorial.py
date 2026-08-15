"""Cross frozen Paper-2.5 discovery with Paper-3 K/V disclosure policies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper3_kv_materialization.run_oracle_frontier import (
    Policy,
    SELECTORS,
    _aggregate,
    _append,
    _attach_intervals,
    _examples,
    _generate,
    _intervals,
    _load_checkpoint,
    _materialized_positions,
    _row_metrics,
    _teacher_forced,
    _prompt,
    _write_csv,
)
from pra_hf import PRAConfig, PRAForCausalLM
from pra_hf.output_validation import (
    MATERIALIZATION_BANDS,
    deterministic_answer_metrics,
    fixed_chunks_for_spans,
    selected_span_metrics,
)
from pra_torch.materialization import union_intervals


def _factorial_policies(selector: str):
    return (
        Policy(f"{selector}__whole_parent", "selected_chunks"),
        Policy(f"{selector}__local_atomic", "logical_intervals", 0, 0),
        Policy(f"{selector}__budget_128", "logical_intervals", 64, 64, 128, "equal"),
        Policy(f"{selector}__gist_local", "gist_plus_logical_intervals", 0, 0),
    )


def _plot(rows, output_dir: Path):
    colors = {
        "one_shot": "#32688f",
        "graph_sparse": "#4f8f55",
        "graph_balanced": "#c27a2c",
        "graph_high": "#a34e59",
    }
    markers = {
        "whole_parent": "o",
        "local_atomic": "s",
        "budget_128": "^",
        "gist_local": "D",
    }
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for dataset, axis in zip(("musique", "2wikimultihopqa"), axes):
        for row in rows:
            if row["dataset"] != dataset:
                continue
            selector, materialization = row["condition"].split("__", 1)
            axis.scatter(
                row["materialized_unique_tokens"],
                row["gold_mean_token_logprob"],
                color=colors[selector],
                marker=markers[materialization],
                s=55,
                label=f"{selector}: {materialization}",
            )
        axis.set_title(dataset)
        axis.set_xlabel("Materialized unique K/V tokens")
        axis.set_ylabel("Gold mean-token log probability")
        axis.grid(alpha=0.25)
    handles, labels = axes[1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    figure.legend(unique.values(), unique.keys(), loc="lower center", ncol=4, fontsize=8)
    figure.tight_layout(rect=(0, 0.16, 1, 1))
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"selector_materialization_factorial.{suffix}", dpi=180)
    plt.close(figure)


def run(args):
    device = torch.device(args.device)
    manifest = json.loads(args.discovery.read_text(encoding="utf-8"))
    discovery_rows = manifest["rows"]
    discovery = {
        (row["dataset"], row["example_id"], row["selection"]): row
        for row in discovery_rows
    }
    examples = _examples(args, discovery_rows)
    inherited = json.loads(args.band_selection.read_text(encoding="utf-8"))["selected_bands"]
    bands = {band.name: band for band in MATERIALIZATION_BANDS}
    heldout = json.loads(args.heldout.read_text(encoding="utf-8"))["rows"]
    no_memory = {
        (row["dataset"], row["example_id"]): row
        for row in heldout
        if row["condition"] == "M_none"
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "selector_factorial_checkpoint.jsonl"
    rows = _load_checkpoint(checkpoint)
    completed = {
        (row["dataset"], row["example_id"], row["condition"])
        for row in rows
    }

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        attn_implementation="eager",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    pra = PRAForCausalLM.from_model(
        model,
        tokenizer,
        pra_config=PRAConfig(
            routing_layer=27,
            consumption_layers=tuple(range(28)),
            chunk_tokens=args.parent_tokens,
            selected_fraction=None,
            top_k=1,
            max_direct_context=args.prompt_tokens,
            native_operation_limit=args.native_limit,
            max_materialized_tokens=args.native_limit - args.prompt_tokens,
            context_safety_reserve_tokens=0,
            encoding_block_tokens=args.encoding_tokens,
            reference_device="cpu",
            pin_reference_memory=device.type == "cuda",
            non_blocking_transfer=device.type == "cuda",
        ),
    )

    for example_index, example in enumerate(examples, start=1):
        dataset = example.dataset
        band = bands[inherited[dataset]]
        pending = [
            (selector, policy)
            for selector in SELECTORS
            for policy in _factorial_policies(selector)
            if (dataset, example.example_id, policy.name) not in completed
        ]
        if not pending:
            continue
        pra.clear_references()
        uri = f"benchmark://{dataset}/{example.example_id}"
        pra.add_reference(example.source, uri=uri)
        entry = pra._handle.cache.get(uri)
        if entry is None:
            raise AssertionError("reference cache entry missing after encoding")
        encoded = _prompt(tokenizer, example.question)

        for selector, policy in pending:
            selected_row = discovery[(dataset, example.example_id, selector)]
            selected_spans = [tuple(map(int, span)) for span in selected_row["selected_spans"]]
            evidence_spans = [tuple(map(int, span)) for span in selected_row["evidence_token_spans"]]
            source_tokens = int(selected_row["source_tokens"])
            selected = fixed_chunks_for_spans(
                entry,
                routing_layer=pra.routing_layer,
                selected_spans=selected_spans,
                selection_name=selector,
            )
            interval_plan = (
                _intervals(uri, selected_spans, source_tokens, policy)
                if policy.mode in {"logical_intervals", "gist_plus_logical_intervals"}
                else []
            )
            fixed = _attach_intervals(selected, interval_plan) if interval_plan else selected
            mapped = pra._handle.map_chunk_identities_to_layers([fixed], band.layers)
            pra._handle.pra_config.detail_materialization = policy.mode
            pra._handle.configure_memory_layers(set(band.layers), fixed_selections=mapped)
            positions = _materialized_positions(policy, fixed, interval_plan)
            teacher = _teacher_forced(
                pra,
                tokenizer,
                encoded,
                example.answer,
                device,
                positions,
                evidence_spans,
                band.layers,
            )
            generation = _generate(
                pra.model, tokenizer, encoded, device, args.max_new_tokens
            )
            physical = _row_metrics(teacher["diagnostics_by_layer"], band.layers)
            materialized_spans = (
                [(interval.start, interval.end) for interval in union_intervals(interval_plan)]
                if interval_plan
                else [(hit.logical_start, hit.logical_end) for hit in selected]
            )
            span_metrics = selected_span_metrics(
                materialized_spans, evidence_spans, source_tokens
            )
            unique = int(physical["materialized_unique_tokens"])
            evidence_tokens = int(span_metrics["evidence_kv_tokens"])
            row = {
                "phase": "selector_factorial",
                "dataset": dataset,
                "example_id": example.example_id,
                "question_type": example.question_type,
                "annotated_hops": example.annotated_hops,
                "condition": policy.name,
                "selection_policy": selector,
                "oracle_labels_used": False,
                "materialization_policy": policy.mode,
                "radius_left": policy.radius_left,
                "radius_right": policy.radius_right,
                "kv_budget": policy.kv_budget,
                "materialization_band": band.name,
                "materialization_layers": list(band.layers),
                "logical_source_tokens": source_tokens,
                "encoding_granularity_tokens": args.encoding_tokens,
                "cpu_reference_cache_bytes": source_tokens * int(pra.model.config.num_hidden_layers) * 2 * int(pra.model.config.num_key_value_heads) * int(pra.model.config.head_dim) * 2,
                "gpu_reference_cache_bytes": 0,
                "conceptual_selected_parents": selected_row["conceptual_active_parents"],
                "evidence_recall": selected_row["oracle_evidence_recall"],
                "complete_evidence_recovery": selected_row["complete_evidence_recovery"],
                "annotated_edge_recall": selected_row["annotated_edge_recall"],
                "requested_materialization_tokens": sum(interval.token_count for interval in interval_plan) if interval_plan else sum(hit.selected_token_count for hit in selected),
                "deduplicated_materialization_tokens": unique,
                "materialized_unique_tokens": unique,
                "active_kv_fraction": unique / max(source_tokens, 1),
                "h2d_kv_bytes": physical["native_kv_bytes"],
                "evidence_kv_tokens": evidence_tokens,
                "evidence_source_tokens": int(span_metrics["evidence_source_tokens"]),
                "evidence_coverage": evidence_tokens / max(int(span_metrics["evidence_source_tokens"]), 1),
                "non_evidence_kv_tokens": max(0, unique - evidence_tokens),
                "evidence_density": evidence_tokens / max(unique, 1),
                "reference_answer": example.answer,
                **physical,
                **teacher,
                **generation,
                **deterministic_answer_metrics(generation["generated_answer"], example.answer),
            }
            row.pop("diagnostics_by_layer", None)
            baseline = no_memory[(dataset, example.example_id)]
            row["gold_mean_logprob_delta_vs_none"] = (
                row["gold_mean_token_logprob"] - baseline["gold_mean_token_logprob"]
            )
            _append(checkpoint, row)
            rows.append(row)
            completed.add((dataset, example.example_id, policy.name))
        print(
            f"[factorial {example_index}/{len(examples)}] {dataset} "
            f"{example.example_id} rows={len(pending)}",
            flush=True,
        )

    aggregates = _aggregate(rows)
    artifact = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "protocol": "frozen Paper-2.5 selection x Paper-3 materialization",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "backbone_frozen": True,
        "oracle_labels_available_to_selection_or_materialization": False,
        "examples_per_dataset": args.examples_per_dataset,
        "selectors": list(SELECTORS),
        "materialization_policies": ["whole_parent", "local_atomic", "budget_128", "gist_local"],
        "rows": rows,
        "aggregates": aggregates,
    }
    (args.output_dir / "selector_materialization_factorial.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(
        args.output_dir / "selector_materialization_factorial_rows.csv",
        [{key: value for key, value in row.items() if not isinstance(value, (dict, list))} for row in rows],
    )
    _write_csv(args.output_dir / "selector_materialization_factorial_aggregate.csv", aggregates)
    _plot(aggregates, args.output_dir)
    return artifact


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    external_data = ROOT / "data/.paper2_5_datasets"
    inherited = ROOT / "docs/papers/shared/results/paper2_5_iterative_pra/output_validation"
    output = ROOT / "docs/papers/shared/results/paper3_kv_materialization"
    parser.add_argument("--phase", default="factorial")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--examples-per-dataset", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--native-limit", type=int, default=4096)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--encoding-tokens", type=int, default=256)
    parser.add_argument("--parent-tokens", type=int, default=32)
    parser.add_argument("--musique-dev", type=Path, default=external_data / "musique/data/musique_ans_v1.0_dev.jsonl")
    parser.add_argument("--twowiki-dev", type=Path, default=external_data / "2wiki/dev.json")
    parser.add_argument("--discovery", type=Path, default=inherited / "gate3_discovery_selections.json")
    parser.add_argument("--band-selection", type=Path, default=inherited / "gate3_materialization_band_selection.json")
    parser.add_argument("--heldout", type=Path, default=output / "oracle_frontier_heldout.json")
    parser.add_argument("--output-dir", type=Path, default=output)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"rows": len(result["rows"])}, indent=2))
