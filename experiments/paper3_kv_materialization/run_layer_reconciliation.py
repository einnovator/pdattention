"""Reconcile PRA consumer placement under corrected native-K/V transport."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper3_kv_materialization.run_oracle_frontier import (
    Policy,
    _annotation_geometry,
    _attach_intervals,
    _attention_metrics,
    _examples,
    _intervals,
    _materialized_positions,
    _prompt,
    _row_metrics,
)
from pra_hf import PRAConfig, PRAForCausalLM
from pra_hf.output_validation import fixed_chunks_for_spans


@dataclass(frozen=True)
class PlacementCondition:
    """One consumer profile crossed with one fixed materialization geometry."""

    profile: str
    layers: tuple[int, ...]
    geometry: str
    radius: int | None = None

    @property
    def name(self) -> str:
        return f"{self.geometry}__{self.profile}"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _load(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _schedules(layer_count: int) -> dict[str, tuple[int, ...]]:
    def last(count: int) -> tuple[int, ...]:
        return tuple(range(max(0, layer_count - count), layer_count))

    def even(count: int) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    round(index * (layer_count - 1) / (count - 1))
                    for index in range(count)
                }
            )
        )

    middle4 = (layer_count - 4) // 2
    return {
        "all": tuple(range(layer_count)),
        "last_24": last(24),
        "last_20": last(20),
        "last_16": last(16),
        "last_14": last(14),
        "last_12": last(12),
        "last_8": last(8),
        "last_4": last(4),
        "last_1": last(1),
        "early_4": tuple(range(4)),
        "middle_4": tuple(range(middle4, middle4 + 4)),
        "even_4": even(4),
        "even_8": even(8),
    }


def _conditions(layer_count: int) -> tuple[PlacementCondition, ...]:
    schedules = _schedules(layer_count)
    rows = [
        PlacementCondition(name, layers, "whole_parent")
        for name, layers in schedules.items()
    ]
    for profile in ("all", "last_14", "last_8", "even_8"):
        layers = schedules[profile]
        rows.extend(
            (
                PlacementCondition(profile, layers, "exact_core", 0),
                PlacementCondition(profile, layers, "expanded_window", 2),
                PlacementCondition(profile, layers, "full_selected_record"),
            )
        )
    return tuple(rows)


def _residual_divergence(
    current: list[torch.Tensor], baseline: list[torch.Tensor]
) -> tuple[float, float, list[float]]:
    values = []
    for left, right in zip(current, baseline):
        left = left.float()
        right = right.float()
        values.append(
            float(
                (1.0 - F.cosine_similarity(left, right, dim=-1).mean())
                .clamp_min(0)
                .cpu()
            )
        )
    return sum(values) / max(len(values), 1), values[-1], values


@torch.no_grad()
def _score(
    pra,
    tokenizer,
    encoded,
    answer: str,
    device: torch.device,
    *,
    position_offset: int,
    memory_positions,
    evidence_spans,
    layers,
):
    answer_ids = tokenizer(
        answer, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    prompt_tokens = int(encoded.input_ids.shape[1])
    full_ids = torch.cat((encoded.input_ids.to(device), answer_ids), dim=1)
    prediction_positions = list(range(prompt_tokens - 1, full_ids.shape[1] - 1))
    positions = torch.arange(
        position_offset,
        position_offset + full_ids.shape[1],
        device=device,
    ).unsqueeze(0)
    pra._handle.set_attention_diagnostics(bool(layers))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = pra.model(
        input_ids=full_ids,
        attention_mask=torch.ones_like(full_ids),
        position_ids=positions,
        output_hidden_states=True,
        use_cache=False,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    logits = output.logits[:, prediction_positions, :].float()
    logprobs = F.log_softmax(logits, dim=-1).gather(
        -1, answer_ids.unsqueeze(-1)
    ).squeeze(-1)[0]
    first = logits[0, 0]
    target = int(answer_ids[0, 0])
    hidden = [
        state[0, prediction_positions, :].float().mean(dim=0).cpu()
        for state in output.hidden_states[1:]
    ]
    attention = (
        _attention_metrics(
            pra, prediction_positions, memory_positions, evidence_spans, layers
        )
        if layers
        else {
            "memory_attention_mass": None,
            "evidence_attention_mass": None,
            "non_evidence_attention_mass": None,
            "attention_entropy": None,
        }
    )
    diagnostics = pra._handle.diagnostics_by_layer()
    pra._handle.set_attention_diagnostics(False)
    return {
        "gold_mean_token_logprob": float(logprobs.mean().cpu()),
        "gold_sequence_logprob": float(logprobs.sum().cpu()),
        "gold_first_token_rank": int((first > first[target]).sum().item()) + 1,
        "gold_first_token_probability": float(first.softmax(-1)[target].cpu()),
        "teacher_forced_seconds": time.perf_counter() - started,
        "hidden": hidden,
        "diagnostics": diagnostics,
        **attention,
    }


def run(args) -> dict:
    device = torch.device(args.device)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "layer_reconciliation_checkpoint.jsonl"
    rows = _load(checkpoint)
    completed = {(row["dataset"], row["example_id"], row["condition"]) for row in rows}

    discovery_payload = json.loads(args.discovery.read_text(encoding="utf-8"))
    discovery_rows = discovery_payload["rows"]
    discovery = {
        (row["dataset"], row["example_id"], row["selection"]): row
        for row in discovery_rows
    }
    example_args = SimpleNamespace(
        musique_dev=args.musique_dev,
        twowiki_dev=args.twowiki_dev,
        phase="heldout",
        examples_per_dataset=args.examples_per_dataset,
    )
    examples = _examples(example_args, discovery_rows)

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
    layer_count = int(model.config.num_hidden_layers)
    conditions = _conditions(layer_count)
    pra = PRAForCausalLM.from_model(
        model,
        tokenizer,
        pra_config=PRAConfig(
            routing_layer=layer_count - 1,
            consumption_layers=tuple(range(layer_count)),
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
        canonical = discovery.get((example.dataset, example.example_id, "oracle_evidence"))
        if canonical is None:
            canonical = _annotation_geometry(tokenizer, example)
        evidence_spans = [tuple(map(int, span)) for span in canonical["evidence_token_spans"]]
        source_tokens = int(canonical["source_tokens"])
        uri = f"benchmark://{example.dataset}/{example.example_id}"
        pra.clear_references()
        pra.add_reference(example.source, uri=uri)
        entry = pra._handle.cache.get(uri)
        if entry is None:
            raise AssertionError("reference cache entry missing")
        selected_parent = fixed_chunks_for_spans(
            entry,
            routing_layer=pra.routing_layer,
            selected_spans=evidence_spans,
            selection_name="oracle_evidence",
        )
        encoded = _prompt(tokenizer, example.question)
        pra._handle.configure_memory_layers(set())
        baseline = _score(
            pra,
            tokenizer,
            encoded,
            example.answer,
            device,
            position_offset=source_tokens,
            memory_positions=[],
            evidence_spans=evidence_spans,
            layers=(),
        )
        for condition in conditions:
            key = (example.dataset, example.example_id, condition.name)
            if key in completed:
                continue
            if condition.geometry in {"exact_core", "expanded_window"}:
                policy = Policy(
                    condition.geometry,
                    "logical_intervals",
                    condition.radius,
                    condition.radius,
                )
                intervals = _intervals(uri, evidence_spans, source_tokens, policy)
                selected = _attach_intervals(selected_parent, intervals)
                memory_positions = _materialized_positions(policy, selected, intervals)
                pra._handle.pra_config.detail_materialization = "logical_intervals"
            else:
                intervals = []
                selected = selected_parent
                memory_positions = _materialized_positions(
                    Policy(condition.geometry, "native_kv"), selected, intervals
                )
                pra._handle.pra_config.detail_materialization = "native_kv"
            mapped = pra._handle.map_chunk_identities_to_layers(
                [selected], condition.layers
            )
            pra._handle.configure_memory_layers(
                set(condition.layers), fixed_selections=mapped
            )
            scored = _score(
                pra,
                tokenizer,
                encoded,
                example.answer,
                device,
                position_offset=source_tokens,
                memory_positions=memory_positions,
                evidence_spans=evidence_spans,
                layers=condition.layers,
            )
            mean_divergence, final_divergence, by_layer = _residual_divergence(
                scored.pop("hidden"), baseline["hidden"]
            )
            physical = _row_metrics(scored.pop("diagnostics"), condition.layers)
            row = {
                "dataset": example.dataset,
                "example_id": example.example_id,
                "condition": condition.name,
                "profile": condition.profile,
                "geometry": condition.geometry,
                "radius": condition.radius,
                "consumer_layers": list(condition.layers),
                "consumer_layer_count": len(condition.layers),
                "consumer_layer_fraction": len(condition.layers) / layer_count,
                "address_layers": [layer_count - 1],
                "detail_kv_layers": list(range(layer_count)),
                "routing_layers": [layer_count - 1],
                "position_mode": "corrected",
                "query_position_offset": source_tokens,
                "source_tokens": source_tokens,
                "residual_divergence_mean": mean_divergence,
                "residual_divergence_final": final_divergence,
                "residual_divergence_by_layer": by_layer,
                "gold_mean_logprob_delta_vs_none": (
                    scored["gold_mean_token_logprob"]
                    - baseline["gold_mean_token_logprob"]
                ),
                **physical,
                **scored,
            }
            _append(checkpoint, row)
            rows.append(row)
            completed.add(key)
        print(
            f"[{example_index}/{len(examples)}] {example.dataset} "
            f"{example.example_id} complete",
            flush=True,
        )

    flat_rows = [
        {
            **row,
            "consumer_layers": json.dumps(row["consumer_layers"]),
            "address_layers": json.dumps(row["address_layers"]),
            "detail_kv_layers": json.dumps(row["detail_kv_layers"]),
            "routing_layers": json.dumps(row["routing_layers"]),
            "residual_divergence_by_layer": json.dumps(
                row["residual_divergence_by_layer"]
            ),
        }
        for row in rows
    ]
    _write_csv(output / "layer_reconciliation_rows.csv", flat_rows)
    manifest = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "examples_per_dataset": args.examples_per_dataset,
        "datasets": ["musique", "2wikimultihopqa"],
        "conditions": [
            {
                "name": condition.name,
                "profile": condition.profile,
                "layers": list(condition.layers),
                "geometry": condition.geometry,
                "radius": condition.radius,
            }
            for condition in conditions
        ],
        "transport": {
            "position": "source-relative reference; query starts after source extent",
            "detail": "destination-layer native post-RoPE K/V",
            "head_layout": "physical GQA heads",
            "attention": "shared local-plus-memory softmax",
            "lifetime": "request prefill/teacher-forced scope",
        },
    }
    (output / "layer_reconciliation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {"rows": len(rows), "examples": len(examples)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    external = Path(r"D:\git\rd\pdattention-iter-gist\data\.paper2_5_datasets")
    inherited = ROOT / "docs/papers/shared/results/paper2_5_iterative_pra/output_validation"
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--examples-per-dataset", type=int, default=16)
    parser.add_argument("--native-limit", type=int, default=4096)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--encoding-tokens", type=int, default=256)
    parser.add_argument("--parent-tokens", type=int, default=32)
    parser.add_argument(
        "--musique-dev",
        type=Path,
        default=external / "musique/data/musique_ans_v1.0_dev.jsonl",
    )
    parser.add_argument(
        "--twowiki-dev", type=Path, default=external / "2wiki/dev.json"
    )
    parser.add_argument(
        "--discovery",
        type=Path,
        default=inherited / "gate3_discovery_selections.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "docs/papers/shared/results/paper3_kv_materialization/layer_reconciliation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
