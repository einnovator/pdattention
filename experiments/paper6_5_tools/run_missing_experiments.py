"""Run large-catalog model choice and typed progressive-disclosure gates."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import string
import sys
import time
import urllib.request
from pathlib import Path
from typing import Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper6_5_tools.prepare_missing_experiments import OUTPUT
from experiments.paper6_5_tools.prepare_progressive_disclosure import TOOL_CASES
from experiments.paper6_5_tools.run_progressive_disclosure import (
    FrozenCapabilityModel,
    _run_tool_condition,
)
from pra_hf.agent_execution import SafeToolExecutor
from pra_hf.agent_resources import AgentResource
from pra_hf.context_records import tool_definition_record


PRIMARY_POLICIES = ("A0_bm25", "A1_fused", "A2_raw_union", "A3_diversity_union")
FULL_VIEW_POLICIES = ("A1_fused", "A2_raw_union")
PROGRESSIVE_POLICIES = ("A1_fused", "A2_raw_union", "A3_diversity_union")
PROGRESSIVE_SIZES = (512, 2048, 8192)
PROGRESSIVE_BUDGETS = (4, 8, 12, 16, 20, 24, 32)
CONDITION_MAP = {
    "T0_full_all": "C0_full_all_one_pass",
    "T1_selection_only": "C1_compact_only",
    "T2_selection_to_full": "C2_compact_to_full",
    "T3_oracle_full": "C3_oracle_to_full",
}
LABEL_PATTERN = re.compile(r"(?<!\d)(\d+)(?!\d)")
HF_LABELS = tuple(string.ascii_uppercase + string.ascii_lowercase)


class OllamaLabelModel:
    """Persistent Q4 CPU backend for bounded one-label capability choice."""

    def __init__(self, model: str = "qwen3:0.6b") -> None:
        self.model_name = model
        self.device = "cpu:q4_ollama"

    def choose(self, prompt: str, names: Sequence[str]) -> tuple[str, dict[str, object]]:
        bound = (
            prompt
            + "\n\nVALID LABELS:\n"
            + "\n".join(f"{index} = {name}" for index, name in enumerate(names))
            + "\nReturn only the integer label of the best candidate.\nLABEL:"
        )
        payload = {
            "model": self.model_name, "stream": False, "keep_alive": "30m", "raw": True,
            "prompt": bound,
            "options": {"temperature": 0, "num_predict": 4, "num_gpu": 0, "num_ctx": 8192, "stop": ["\n"]},
        }
        request = urllib.request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.loads(response.read())
        elapsed = time.perf_counter() - started
        text = str(result.get("response", ""))
        match = LABEL_PATTERN.search(text)
        label = int(match.group(1)) if match else -1
        chosen = names[label] if 0 <= label < len(names) else ""
        return chosen, {
            "prompt_tokens": int(result.get("prompt_eval_count", 0)),
            "generated_tokens": int(result.get("eval_count", 0)),
            "batch_wall_seconds": elapsed,
            "amortized_wall_seconds": elapsed,
            "ttft_seconds_upper_bound": elapsed,
            "batch_size": 1,
            "ttft_method": "ollama_total_call_upper_bound",
            "selected_mean_log_probability": "",
            "choice_margin": "",
            "raw_label_response": text.replace("\n", "\\n"),
        }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_checkpoint(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _resource(row: Mapping[str, object]) -> AgentResource:
    return AgentResource(
        uri=str(row["uri"]), kind="tool", namespace=str(row["namespace"]),
        name=str(row["name"]), version=str(row["version"]),
        description=str(row["description"]),
        content=json.dumps(row["provider_schema"], sort_keys=True),
        side_effect_class=str(row["side_effect"]), tenant_id=str(row["tenant_id"]),
        metadata={"signature": row["signature"], "scaled_catalog": True},
    )


def _choice_prompt(query: str, payload: str, capability_type: str) -> str:
    """Build the common candidate-comparison prompt before label binding."""

    return (
        f"Choose the single {capability_type} that best fits the request. "
        "The bounded records below are the only valid candidates.\n\n"
        f"REQUEST:\n{query}\n\nCANDIDATE RECORDS:\n{payload}"
    )


def _batch_choose(
    model,
    prompts: Sequence[str],
    *,
    candidates: Sequence[Sequence[str]],
    batch_size: int,
) -> list[tuple[str, dict[str, object]]]:
    """Choose among valid identities from one frozen next-label distribution."""

    if isinstance(model, OllamaLabelModel):
        return [model.choose(prompt, names) for prompt, names in zip(prompts, candidates)]

    tokenizer = model.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    rendered_prompts = []
    label_ids = []
    for prompt, names in zip(prompts, candidates):
        if len(names) > len(HF_LABELS):
            raise ValueError(f"At most {len(HF_LABELS)} candidates are supported by label binding.")
        labels = []
        bound_labels = HF_LABELS[: len(names)]
        for label in bound_labels:
            ids = tokenizer.encode(label, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"Candidate label {label!r} is not one tokenizer token.")
            labels.append(ids[0])
        label_ids.append(labels)
        prompt = (
            prompt
            + "\n\nVALID LABELS:\n"
            + "\n".join(f"{label} = {name}" for label, name in zip(bound_labels, names))
            + "\nReturn only the single-character label of the best candidate.\nLABEL:"
        )
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        rendered_prompts.append(rendered)
    results = []
    elapsed_total = 0.0
    for start in range(0, len(rendered_prompts), batch_size):
        rendered = rendered_prompts[start : start + batch_size]
        inputs = tokenizer(rendered, return_tensors="pt", padding=True).to(model.device)
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        started = time.perf_counter()
        with torch.inference_mode():
            logits = model.model(
                **inputs, logits_to_keep=1,
            ).logits[:, -1].float()
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        elapsed = time.perf_counter() - started
        elapsed_total += elapsed
        for local_index in range(len(rendered)):
            global_index = start + local_index
            ids = label_ids[global_index]
            values = logits[local_index, ids]
            probabilities = values.log_softmax(dim=0)
            selected_index = int(values.argmax().item())
            alternatives = [float(value) for index, value in enumerate(values.tolist()) if index != selected_index]
            results.append((candidates[global_index][selected_index], {
                "prompt_tokens": int(inputs.attention_mask[local_index].sum().item()),
                "generated_tokens": 1,
                "batch_wall_seconds": elapsed,
                "amortized_wall_seconds": elapsed / len(rendered),
                "ttft_seconds_upper_bound": elapsed,
                "batch_size": len(rendered),
                "ttft_method": "single_next_label_forward",
                "selected_mean_log_probability": float(probabilities[selected_index].item()),
                "choice_margin": float(values[selected_index].item()) - max(alternatives, default=float(values[selected_index].item())),
            }))
    return results


def _payload(view_by, palette, view: str) -> str:
    return "\n".join(
        str(view_by[(palette["catalog_size"], palette["seed"], name)][f"{view}_payload"])
        for name in palette["candidate_names"]
    )


def _run_palette_choice(model, palettes, view_by, args, checkpoint):
    rows = list(checkpoint.get("palette_rows", ()))
    completed = {
        (int(row["catalog_size"]), int(row["seed"]), row["query_id"], row["policy"],
         int(row["max_candidates"]), row["candidate_view"])
        for row in rows
    }
    pending = []
    prompts = []
    for palette in palettes:
        policy = str(palette["policy"])
        replicated_key = int(palette["seed"]) == 11 or int(palette["catalog_size"]) == 8192
        is_primary = policy in PRIMARY_POLICIES and replicated_key
        is_agreement_control = (
            policy == "A4_agreement_union"
            and int(palette["catalog_size"]) == 8192
            and int(palette["max_candidates"]) in (10, 32)
        )
        if not (is_primary or is_agreement_control):
            continue
        key = (int(palette["catalog_size"]), int(palette["seed"]), palette["query_id"],
               policy, int(palette["max_candidates"]), "compact")
        if key in completed:
            continue
        pending.append(palette)
        prompts.append(_choice_prompt(str(palette["query"]), _payload(view_by, palette, "selection"), "tool"))
    for offset in range(0, len(pending), args.checkpoint_batch):
        batch_palettes = pending[offset : offset + args.checkpoint_batch]
        values = _batch_choose(
            model,
            prompts[offset : offset + args.checkpoint_batch],
            candidates=[row["candidate_names"] for row in batch_palettes],
            batch_size=args.batch_size,
        )
        for palette, (generated, costs) in zip(batch_palettes, values):
            chosen = generated
            candidate = next((row for row in palette["candidates"] if row["name"] == chosen), None)
            target_in = int(palette["target_in_palette"])
            correct = int(chosen == palette["target_name"])
            rows.append({
                "catalog_size": palette["catalog_size"], "seed": palette["seed"],
                "scoring_device": str(model.device),
                "query_id": palette["query_id"], "policy": palette["policy"],
                "max_candidates": palette["max_candidates"], "candidate_view": "compact",
                "target_name": palette["target_name"], "target_in_palette": target_in,
                "chosen_tool": chosen, "choice_correct": correct,
                "conditional_choice_denominator": target_in,
                "candidate_count": palette["candidate_count"],
                "candidate_names": "|".join(palette["candidate_names"]),
                "candidate_sources": "" if candidate is None else "|".join(row["channel"] for row in candidate["sources"]),
                "candidate_ranks": "" if candidate is None else "|".join(f"{row['channel']}:{row['rank']}" for row in candidate["sources"]),
                "channel_agreement": 0 if candidate is None else candidate["channel_agreement"],
                "unsafe_candidates": "|".join(palette["unsafe_candidates"]),
                "unsafe_exposure": int(bool(palette["unsafe_candidates"])),
                "unsafe_choice": int(chosen in set(palette["unsafe_candidates"])),
                "useful_candidates": "|".join(palette["useful_candidates"]),
                "selection_view_tokens": sum(int(view_by[(palette["catalog_size"], palette["seed"], name)]["selection_tokens"]) for name in palette["candidate_names"]),
                "all_candidate_full_tokens": sum(int(view_by[(palette["catalog_size"], palette["seed"], name)]["full_tokens"]) for name in palette["candidate_names"]),
                **costs,
                "generated_text": json.dumps({"capability": generated}, separators=(",", ":")),
            })
        checkpoint["palette_rows"] = rows
        _write_checkpoint(args.output / "missing_experiments_checkpoint.json", checkpoint)
        print(f"palette choice {min(offset + args.checkpoint_batch, len(pending))}/{len(pending)}", flush=True)
    return rows


def _run_full_view_choice(model, palettes, view_by, args, checkpoint):
    rows = list(checkpoint.get("full_view_rows", ()))
    completed = {(int(row["catalog_size"]), int(row["seed"]), row["query_id"], row["policy"], int(row["max_candidates"])) for row in rows}
    selected = [row for row in palettes if int(row["catalog_size"]) in args.view_sizes
                and int(row["max_candidates"]) in args.progressive_budgets
                and row["policy"] in FULL_VIEW_POLICIES
                and row["query_id"] in set(args.full_view_queries)
                and (int(row["seed"]) == 11 or (
                    int(row["catalog_size"]) == 8192
                            and int(row["max_candidates"]) in (8, 16, 24, 32)
                ))]
    pending = [row for row in selected if (int(row["catalog_size"]), int(row["seed"]), row["query_id"], row["policy"], int(row["max_candidates"])) not in completed]
    prompts = [_choice_prompt(str(row["query"]), _payload(view_by, row, "full"), "tool") for row in pending]
    for offset in range(0, len(pending), args.checkpoint_batch):
        batch_palettes = pending[offset : offset + args.checkpoint_batch]
        values = _batch_choose(
            model, prompts[offset : offset + args.checkpoint_batch],
            candidates=[row["candidate_names"] for row in batch_palettes],
            batch_size=args.batch_size,
        )
        for palette, (generated, costs) in zip(batch_palettes, values):
            chosen = generated
            rows.append({
                "catalog_size": palette["catalog_size"], "seed": palette["seed"],
                "scoring_device": str(model.device),
                "query_id": palette["query_id"], "policy": palette["policy"],
                "max_candidates": palette["max_candidates"], "candidate_view": "full_all",
                "target_name": palette["target_name"], "target_in_palette": palette["target_in_palette"],
                "chosen_tool": chosen, "choice_correct": int(chosen == palette["target_name"]),
                "conditional_choice_denominator": int(palette["target_in_palette"]),
                "candidate_count": palette["candidate_count"],
                "selection_view_tokens": sum(int(view_by[(palette["catalog_size"], palette["seed"], name)]["selection_tokens"]) for name in palette["candidate_names"]),
                "all_candidate_full_tokens": sum(int(view_by[(palette["catalog_size"], palette["seed"], name)]["full_tokens"]) for name in palette["candidate_names"]),
                "unsafe_exposure": int(bool(palette["unsafe_candidates"])),
                "unsafe_choice": int(chosen in set(palette["unsafe_candidates"])),
                **costs,
                "generated_text": json.dumps({"capability": generated}, separators=(",", ":")),
            })
        checkpoint["full_view_rows"] = rows
        _write_checkpoint(args.output / "missing_experiments_checkpoint.json", checkpoint)
        print(f"full-view choice {min(offset + args.checkpoint_batch, len(pending))}/{len(pending)}", flush=True)
    return rows


def _run_progressive(model, palettes, view_by, args, checkpoint):
    rows = list(checkpoint.get("progressive_rows", ()))
    completed = {(int(row["catalog_size"]), int(row["seed"]), row["query_id"], row["policy"], int(row["max_candidates"]), row["condition"]) for row in rows}
    case_arguments = {row.query_id: dict(row.expected_arguments) for row in TOOL_CASES}
    selected = [row for row in palettes if int(row["catalog_size"]) in args.progressive_sizes
                and int(row["max_candidates"]) in args.progressive_budgets
                and row["policy"] in PROGRESSIVE_POLICIES
                and row["query_id"] in set(args.progressive_queries)
                and (
                    (row["policy"] in FULL_VIEW_POLICIES and (
                        int(row["seed"]) == 11 or (
                            row["policy"] == "A3_diversity_union"
                            and int(row["catalog_size"]) == 8192
                            and int(row["max_candidates"]) in (8, 16, 24, 32)
                        )
                    ))
                    or (
                        row["policy"] == "A3_diversity_union"
                        and int(row["catalog_size"]) == 8192
                        and int(row["max_candidates"]) in (8, 16, 24, 32)
                    )
                )]
    for index, palette in enumerate(selected, start=1):
        all_names = tuple(dict.fromkeys((*palette["candidate_names"], palette["target_name"])))
        resources_by_name = {
            name: _resource(view_by[(palette["catalog_size"], palette["seed"], name)])
            for name in all_names
        }
        records_by_name = {name: tool_definition_record(resource) for name, resource in resources_by_name.items()}
        executor = SafeToolExecutor(
            tuple(resources_by_name.values()),
            {resource.uri: (lambda _arguments, _observations: {"ok": True}) for resource in resources_by_name.values()},
        )
        case = {
            "query_id": palette["query_id"], "query": palette["query"],
            "target_name": palette["target_name"],
            "expected_arguments": case_arguments[palette["query_id"]],
        }
        for source_condition, condition in CONDITION_MAP.items():
            key = (int(palette["catalog_size"]), int(palette["seed"]), palette["query_id"], palette["policy"], int(palette["max_candidates"]), condition)
            if key in completed:
                continue
            metrics = _run_tool_condition(
                model, source_condition, case, tuple(palette["candidate_names"]),
                resources_by_name, records_by_name, executor,
            )
            full_tokens = sum(int(view_by[(palette["catalog_size"], palette["seed"], name)]["full_tokens"]) for name in palette["candidate_names"])
            total = int(metrics["total_capability_tokens"])
            rows.append({
                "catalog_size": palette["catalog_size"], "seed": palette["seed"],
                "scoring_device": str(model.device),
                "query_id": palette["query_id"], "policy": palette["policy"],
                "max_candidates": palette["max_candidates"], "condition": condition,
                "target_name": palette["target_name"], "target_in_palette": palette["target_in_palette"],
                "candidate_count": palette["candidate_count"],
                "all_candidate_full_tokens": full_tokens,
                "disclosure_ratio_a": metrics["phase_a_capability_tokens"] / max(full_tokens, 1),
                "disclosure_ratio_total": total / max(full_tokens, 1),
                "full_schema_tokens_avoided": max(full_tokens - int(metrics["phase_b_capability_tokens"]), 0),
                "full_schemas_materialized": 1 if condition in {"C2_compact_to_full", "C3_oracle_to_full"} else int(palette["candidate_count"]) if condition == "C0_full_all_one_pass" else 0,
                "unsafe_exposure": int(bool(palette["unsafe_candidates"])),
                **metrics,
            })
            checkpoint["progressive_rows"] = rows
            _write_checkpoint(args.output / "missing_experiments_checkpoint.json", checkpoint)
        print(f"progressive palette {index}/{len(selected)}", flush=True)
    return rows


def _derive_jit(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    by = {(int(row["catalog_size"]), int(row["seed"]), row["query_id"], row["policy"], int(row["max_candidates"]), row["condition"]): row for row in rows}
    workflows = {
        "account_repair": ("get_user-h2-en-1", "update_user-h4-en-1"),
        "document_delivery": ("search_document-h1-en-1", "export_document-h2-en-1"),
    }
    output = []
    for seed in (11, 23, 37, 53, 71):
        for policy in PROGRESSIVE_POLICIES:
            for budget in (8, 16, 24, 32):
                for workflow, query_ids in workflows.items():
                    step_rows = [by.get((8192, seed, query_id, policy, budget, "C2_compact_to_full")) for query_id in query_ids]
                    if any(row is None for row in step_rows):
                        continue
                    workflow_success = int(all(int(row["task_success"]) for row in step_rows))
                    for step, row in enumerate(step_rows, start=1):
                        output.append({
                            "catalog_size": 8192, "seed": seed, "workflow_id": workflow,
                            "step": step, "query_id": row["query_id"], "policy": policy,
                            "max_candidates": budget,
                            "target_in_palette": row["target_in_palette"],
                            "conditional_choice_correct": row["capability_choice_correct"],
                            "tool_call_acceptance": row["execution_acceptance"],
                            "wrong_tool_proposal": row["wrong_tool_choice"],
                            "unsafe_tool_proposal": row["unsafe_tool_choice"],
                            "host_rejection": row["host_rejection"],
                            "step_success": row["task_success"], "workflow_success": workflow_success,
                            "mean_steps": len(step_rows), "replans": 0, "retries": row["retry_count"],
                            "capability_tokens": row["total_capability_tokens"],
                            "full_schemas_materialized": row["full_schemas_materialized"],
                            "resolver_latency_seconds": row["materialization_seconds"],
                            "model_latency_seconds": row["wall_clock_seconds"],
                        })
    return output


def run(args: argparse.Namespace) -> None:
    palettes = _read_jsonl(args.output / "large_catalog_palettes.jsonl")
    views = _read_jsonl(args.output / "progressive_tool_views.jsonl")
    view_by = {(int(row["catalog_size"]), int(row["seed"]), row["name"]): row for row in views}
    checkpoint_path = args.output / "missing_experiments_checkpoint.json"
    checkpoint = {} if args.fresh or not checkpoint_path.exists() else json.loads(checkpoint_path.read_text(encoding="utf-8"))
    model = (
        OllamaLabelModel(args.ollama_model)
        if args.backend == "ollama"
        else FrozenCapabilityModel(torch.device(args.device))
    )
    if args.backend == "ollama" and "progressive" in args.phases:
        raise ValueError("The progressive execution phase requires --backend hf.")
    palette_rows = (
        _run_palette_choice(model, palettes, view_by, args, checkpoint)
        if "palette" in args.phases else list(checkpoint.get("palette_rows", ()))
    )
    full_rows = (
        _run_full_view_choice(model, palettes, view_by, args, checkpoint)
        if "full_view" in args.phases else list(checkpoint.get("full_view_rows", ()))
    )
    progressive_rows = (
        _run_progressive(model, palettes, view_by, args, checkpoint)
        if "progressive" in args.phases else list(checkpoint.get("progressive_rows", ()))
    )
    _write_csv(args.output / "large_catalog_palette_choice.csv", [*palette_rows, *full_rows])
    _write_csv(args.output / "progressive_tool_disclosure.csv", progressive_rows)
    _write_csv(args.output / "large_catalog_progressive_disclosure.csv", progressive_rows)
    _write_csv(args.output / "large_catalog_jit.csv", _derive_jit(progressive_rows))
    _write_csv(args.output / "channel_agreement_ablation.csv", [
        row for row in palette_rows
        if row["policy"] in {"A2_raw_union", "A4_agreement_union"}
        and int(row["catalog_size"]) == 8192 and int(row["max_candidates"]) == 10
    ])
    manifest = {
        "device": args.device, "palette_choice_rows": len(palette_rows),
        "phases_executed": list(args.phases),
        "full_view_choice_rows": len(full_rows), "progressive_rows": len(progressive_rows),
        "jit_rows": len(_derive_jit(progressive_rows)),
        "batch_choice_note": (
            "Palette and matched full-view phases use raw Q4 one-label generation; "
            "progressive execution uses exact FP16 structured calls."
        ),
        "protocol_backends": {
            "palette_and_full_view": "Ollama qwen3:0.6b Q4 raw one-label generation",
            "progressive_execution": "Hugging Face Qwen/Qwen3-0.6B exact FP16 CUDA",
        },
        "generation": {"do_sample": False, "temperature": 0},
        "backend": args.backend,
        "model_artifact": args.ollama_model if args.backend == "ollama" else "Qwen/Qwen3-0.6B exact local revision",
        "replication_design": {
            "full_grid_seed": 11,
            "palette_five_seed_keys": {
                "catalog_size": 8192,
                "max_candidates": [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48],
            },
            "full_view_five_seed_keys": {"catalog_size": 8192, "max_candidates": [8, 16, 24, 32]},
            "progressive_and_jit_five_seed_keys": {
                "catalog_size": 8192,
                "policy": "A3_diversity_union",
                "max_candidates": [8, 16, 24, 32],
            },
        },
    }
    (args.output / "missing_experiments_run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backend", choices=("hf", "ollama"), default="hf")
    parser.add_argument("--ollama-model", default="qwen3:0.6b")
    parser.add_argument("--phases", nargs="+", choices=("palette", "full_view", "progressive"), default=("palette", "full_view", "progressive"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--checkpoint-batch", type=int, default=64)
    parser.add_argument("--view-sizes", nargs="+", type=int, default=(128, 512, 2048, 8192))
    parser.add_argument("--progressive-sizes", nargs="+", type=int, default=PROGRESSIVE_SIZES)
    parser.add_argument("--progressive-budgets", nargs="+", type=int, default=PROGRESSIVE_BUDGETS)
    parser.add_argument(
        "--progressive-queries", nargs="+",
        default=tuple(row.query_id for row in TOOL_CASES[:4]),
        help="Frozen stratified cohort for expensive full tool-call generation.",
    )
    parser.add_argument(
        "--full-view-queries", nargs="+",
        default=tuple(row.query_id for row in TOOL_CASES[:4]),
        help="Frozen stratified subset used for the expensive full-schema choice comparison.",
    )
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
