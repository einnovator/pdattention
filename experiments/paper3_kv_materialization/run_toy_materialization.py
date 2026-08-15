"""Run oracle-only physical K/V disclosure on frozen controlled decoders.

Each source is encoded once as one contextual parent. The parent identity is
the only routing candidate, while explicit logical intervals vary the physical
native K/V made visible at one consumer layer. Thus routing recall is fixed at
one and cannot explain differences between disclosure conditions.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Iterable

import torch

from pra_torch.controlled_local_sa import ControlledExample, ControlledTokenizer, controlled_examples
from pra_torch.materialization import LogicalInterval, union_intervals
from pra_torch.model import TinyPRAModel, convert_sa_model_to_pra

from experiments.paper2_5_iterative_pra.run_controlled_local_sa import (
    SEEDS,
    model_config,
    parse_windows,
    window_name,
)
from experiments.paper3_kv_materialization.toy_materialization import (
    ToyPolicy,
    attention_partition,
    label_metrics,
    materialized_positions,
    policy_intervals,
    representation_portability,
    source_and_fact_spans,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT / "docs/papers/shared/results/paper3_kv_materialization/toy_materialization"
)
DEFAULT_CHECKPOINTS = (
    ROOT.parent
    / "pdattention-iter-gist/docs/papers/shared/results/"
    "paper2_5_iterative_pra/controlled_local_sa_v6/checkpoints"
)
PRIMARY_LAYER = 0
PORTABILITY_LAYER = 2
PROFILE_LAYERS = (0, 2, 5)


def policies() -> tuple[ToyPolicy, ...]:
    """Return the frozen T0--T11 ladder plus bounded dispersion controls."""
    values = [
        ToyPolicy("T0_none", "none"),
        ToyPolicy("T1_radius_0", "logical_intervals", radius=0),
        ToyPolicy("T2_radius_2", "logical_intervals", radius=2),
        ToyPolicy("T3_radius_4", "logical_intervals", radius=4),
        ToyPolicy("T4_radius_8", "logical_intervals", radius=8),
        ToyPolicy("T5_radius_16", "logical_intervals", radius=16),
        ToyPolicy("T6_whole_fact", "logical_intervals", whole_fact=True),
        ToyPolicy("T7_whole_parent", "selected_chunks", whole_parent=True),
    ]
    for budget in (12, 24, 48):
        for allocation in (
            "equal",
            "evidence_length_proportional",
            "minimum_core_remainder",
        ):
            values.append(
                ToyPolicy(
                    f"T8_budget_{budget}_{allocation}",
                    "logical_intervals",
                    radius=16,
                    budget=budget,
                    allocation=allocation,
                )
            )
    values.extend(
        [
            ToyPolicy("T9_wrong_exact", "logical_intervals", wrong_memory=True),
            ToyPolicy("T10_native_gist", "native_gist_only"),
            ToyPolicy("T11_gist_exact", "gist_plus_logical_intervals"),
            ToyPolicy("dispersion_1_radius_4", "logical_intervals", radius=4, region_groups=1),
            ToyPolicy("dispersion_2_radius_4", "logical_intervals", radius=4, region_groups=2),
            ToyPolicy("dispersion_4_radius_4", "logical_intervals", radius=4, region_groups=4),
        ]
    )
    return tuple(values)


class StateCapture:
    """Capture final-token residual states and PRA branch outputs by layer."""

    def __init__(self, model: TinyPRAModel) -> None:
        self.model = model
        self.states: dict[int, torch.Tensor] = {}
        self.pra_outputs: dict[int, torch.Tensor] = {}
        self.handles = []
        for block in model.blocks:
            layer = int(block.layer_id)
            self.handles.append(block.register_forward_hook(self._block_hook(layer)))
            attention = getattr(block, "pra_attn", None)
            if attention is not None:
                self.handles.append(attention.register_forward_hook(self._pra_hook(layer)))

    def _block_hook(self, layer: int):
        def hook(_module, _inputs, output):
            self.states[layer] = output[:, -1, :].detach().clone()
        return hook

    def _pra_hook(self, layer: int):
        def hook(_module, _inputs, output):
            self.pra_outputs[layer] = output[:, -1, :].detach().clone()
        return hook

    def reset(self) -> None:
        self.states.clear()
        self.pra_outputs.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def readout(self, layer: int, answer_id: int) -> dict[str, float | int]:
        logits = self.model.head(self.model.ln(self.states[layer]))[0]
        return label_metrics(logits, answer_id)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _append_record(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def _interval_payload(intervals: Iterable[LogicalInterval]) -> list[dict]:
    return [
        {
            "domain": interval.domain,
            "start": interval.start,
            "end": interval.end,
            "evidence_start": interval.evidence_start,
            "evidence_end": interval.evidence_end,
            "score": interval.score,
        }
        for interval in intervals
    ]


def _build_model(
    source: TinyPRAModel, layer: int, device: str
) -> TinyPRAModel:
    cfg = replace(
        source.cfg,
        model_variant="td_layered_pra",
        pra_layer_ids=(int(layer),),
        top_k_references=1,
        top_k_chunks_per_reference=1,
        trigger_threshold=-1.0,
        detail_materialization="logical_intervals",
        max_materialized_memory_tokens=240,
        context_safety_reserve_tokens=0,
        collect_routing_metrics=True,
        collect_per_head_metrics=True,
        chunking_mode="none",
    )
    return convert_sa_model_to_pra(source, cfg).to(device).eval()


def _parent_entry(
    model: TinyPRAModel,
    tokenizer: ControlledTokenizer,
    example: ControlledExample,
    source: tuple[int, ...],
    domain: str,
    device: str,
):
    return model.encode_reference_tokens_to_cache(
        domain,
        source,
        tokenizer,
        device,
        metadata={
            "selection_source": "oracle_parent",
            "oracle_identity_selection": True,
            "gold_spans_absent_from_routing_state": True,
        },
        text=f"controlled parent for {example.example_id}",
        max_chunks=1,
        use_configured_max_chunks=False,
        max_chunk_tokens=len(source),
    )


def _set_condition(
    model: TinyPRAModel,
    entry,
    policy: ToyPolicy,
    intervals: list[LogicalInterval],
) -> None:
    model.clear_pra_cache()
    model.cfg.detail_materialization = policy.mode
    for block in model.blocks:
        attention = getattr(block, "pra_attn", None)
        if attention is not None:
            attention.config.detail_materialization = policy.mode
    entry.metadata.pop("materialization_intervals", None)
    if policy.mode in {"logical_intervals", "gist_plus_logical_intervals"}:
        entry.metadata["materialization_intervals"] = _interval_payload(intervals)
    if policy.mode != "none":
        model.pra_cache.put(entry)


def _evidence_geometry(
    example: ControlledExample,
    fact_spans: dict[str, tuple[int, int]],
    core_spans: dict[str, tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]], set[int], set[int]]:
    evidence_uris = set(example.target_reference_uris)
    evidence_core = [core_spans[uri] for uri in example.target_reference_uris]
    evidence_fact = [fact_spans[uri] for uri in example.target_reference_uris]
    distractor_core = [
        core_spans[reference.uri]
        for reference in example.references
        if reference.uri not in evidence_uris
    ][: len(evidence_core)]
    evidence_positions = {
        position for start, end in evidence_core for position in range(start, end)
    }
    distractor_positions = {
        position
        for reference in example.references
        if reference.uri not in evidence_uris
        for position in range(*fact_spans[reference.uri])
    }
    return (
        evidence_core,
        evidence_fact,
        distractor_core,
        evidence_positions,
        distractor_positions,
    )


def _portability_row(
    model: TinyPRAModel,
    parent,
    tokenizer: ControlledTokenizer,
    example: ControlledExample,
    core_spans: dict[str, tuple[int, int]],
    *,
    layer: int,
    device: str,
) -> dict:
    contextual = []
    isolated = []
    parent_values = parent.layer_memory[layer].chunks[0].token_kv.v
    for reference in example.references:
        if not reference.is_evidence:
            continue
        start, end = core_spans[reference.uri]
        contextual.append(parent_values[:, :, start:end, :])
        entry = model.encode_reference_tokens_to_cache(
            f"{reference.uri}/isolated",
            reference.token_ids,
            tokenizer,
            device,
            metadata={"portability_control": "isolated_fact"},
            max_chunks=1,
            use_configured_max_chunks=False,
        )
        isolated.append(entry.layer_memory[layer].chunks[0].token_kv.v[:, :, 1:-1, :])
    metrics = representation_portability(
        torch.cat(contextual, dim=2), torch.cat(isolated, dim=2)
    )
    return {
        "example_id": example.example_id,
        "depth": example.depth,
        "evidence_regions": len(example.target_reference_uris),
        "evidence_tokens": 3 * len(example.target_reference_uris),
        **metrics,
    }


@torch.no_grad()
def _condition(
    model: TinyPRAModel,
    capture: StateCapture,
    entry,
    example: ControlledExample,
    policy: ToyPolicy,
    intervals: list[LogicalInterval],
    positions: list[int | None],
    evidence_positions: set[int],
    distractor_positions: set[int],
    *,
    layer: int,
    baseline_states: dict[int, torch.Tensor] | None,
    baseline_pra: dict[int, torch.Tensor] | None,
    device: str,
) -> tuple[dict, list[dict], list[dict]]:
    _set_condition(model, entry, policy, intervals)
    capture.reset()
    ids = torch.tensor([example.query_input_ids], dtype=torch.long, device=device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    logits, trace = model.forward_progressive_pra(
        ids, prevent_reference_replay=False
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    final = label_metrics(logits[0, -1], example.answer_id)
    trace_row = trace[0]
    weights = trace_row.get("final_token_memory_weights", ())
    attention, head_rows = attention_partition(
        weights,
        positions,
        evidence_positions=evidence_positions,
        distractor_positions=distractor_positions,
    )
    physical_positions = {position for position in positions if position is not None}
    evidence_tokens = len(physical_positions & evidence_positions)
    distractor_tokens = len(physical_positions & distractor_positions)
    materialized_tokens = int(trace_row["materialized_tokens"])
    trajectory = []
    for stage_layer in sorted(capture.states):
        stage = capture.readout(stage_layer, example.answer_id)
        baseline_margin = (
            label_metrics(
                model.head(model.ln(baseline_states[stage_layer]))[0],
                example.answer_id,
            )["correct_margin"]
            if baseline_states is not None
            else stage["correct_margin"]
        )
        trajectory.append(
            {
                "layer": stage_layer,
                **stage,
                "margin_delta_vs_none": float(stage["correct_margin"]) - float(baseline_margin),
            }
        )
    immediate = next(
        row["margin_delta_vs_none"] for row in trajectory if row["layer"] == layer
    )
    final_delta = trajectory[-1]["margin_delta_vs_none"]
    baseline_state = baseline_states.get(layer) if baseline_states is not None else None
    baseline_branch = baseline_pra.get(layer) if baseline_pra is not None else None
    residual_displacement = (
        float((capture.states[layer] - baseline_state).norm().cpu())
        if baseline_state is not None
        else 0.0
    )
    attention_update_displacement = (
        float((capture.pra_outputs[layer] - baseline_branch).norm().cpu())
        if baseline_branch is not None
        else 0.0
    )
    row = {
        **final,
        **attention,
        "consumer_layer": layer,
        "materialized_tokens": materialized_tokens,
        "layer_token_kv_states": materialized_tokens,
        "evidence_tokens": evidence_tokens,
        "evidence_coverage": evidence_tokens / max(len(evidence_positions), 1),
        "evidence_density": evidence_tokens / max(materialized_tokens, 1),
        "surrounding_non_evidence_tokens": max(
            len(physical_positions) - evidence_tokens - distractor_tokens, 0
        ),
        "distractor_tokens": distractor_tokens,
        "requested_tokens_pre_dedup": sum(
            interval.token_count for interval in intervals
        ),
        "interval_count": len(intervals),
        "cross_shard_interval_count": float(
            trace_row["materialization_metrics"].get(
                "cross_shard_interval_count", 0.0
            )
        ),
        "immediate_margin_delta_vs_none": immediate,
        "final_margin_delta_vs_none": final_delta,
        "later_erased": int(immediate > 0.0 and final_delta <= 0.0),
        "residual_displacement": residual_displacement,
        "attention_update_displacement": attention_update_displacement,
        "latency_seconds": elapsed,
        "selected_reference_count": len(trace_row["selected_reference_uris"]),
        "oracle_identity_selected": int(
            policy.mode == "none"
            or trace_row["selected_reference_uris"] == [entry.uri]
        ),
    }
    return row, trajectory, head_rows


def _load_source(
    checkpoint: Path,
    tokenizer: ControlledTokenizer,
    *,
    window: int | None,
    device: str,
    d_model: int,
    layers: int,
) -> TinyPRAModel:
    cfg = model_config(
        tokenizer,
        window=window,
        device=device,
        d_model=d_model,
        n_layers=layers,
    )
    source = TinyPRAModel(cfg).to(device).eval()
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    source.load_state_dict(state["model"])
    return source


def _run_model(
    *,
    window: int | None,
    seed: int,
    examples: list[ControlledExample],
    tokenizer: ControlledTokenizer,
    checkpoint: Path,
    checkpoint_log: Path,
    completed: set[tuple],
    device: str,
    d_model: int,
    layers: int,
) -> None:
    source = _load_source(
        checkpoint,
        tokenizer,
        window=window,
        device=device,
        d_model=d_model,
        layers=layers,
    )
    model = _build_model(source, PRIMARY_LAYER, device)
    portability_model = _build_model(source, PORTABILITY_LAYER, device)
    capture = StateCapture(model)
    try:
        for example_index, example in enumerate(examples):
            partition = "validation" if example_index < len(examples) // 2 else "heldout"
            source_tokens, fact_spans, core_spans = source_and_fact_spans(example)
            domain = f"controlled-parent://{example.example_id}"
            parent = _parent_entry(
                model, tokenizer, example, source_tokens, domain, device
            )
            geometry = _evidence_geometry(example, fact_spans, core_spans)
            evidence_core, evidence_fact, wrong_core, evidence_set, distractor_set = geometry
            portability_key = ("portability", window_name(window), seed, example.example_id)
            if portability_key not in completed:
                portability_parent = _parent_entry(
                    portability_model,
                    tokenizer,
                    example,
                    source_tokens,
                    domain,
                    device,
                )
                portability = {
                    "window": window_name(window),
                    "seed": seed,
                    "partition": partition,
                    "representation_layer": PORTABILITY_LAYER,
                    **_portability_row(
                        portability_model,
                        portability_parent,
                        tokenizer,
                        example,
                        core_spans,
                        layer=PORTABILITY_LAYER,
                        device=device,
                    ),
                }
                _append_record(
                    checkpoint_log,
                    {"kind": "portability", "key": list(portability_key), "row": portability},
                )
                completed.add(portability_key)

            baseline_states = baseline_pra = None
            for policy in policies():
                if policy.region_groups is not None and example.depth != 4:
                    continue
                key = (
                    "main",
                    window_name(window),
                    seed,
                    example.example_id,
                    policy.name,
                )
                should_record = key not in completed
                if not should_record and policy.name != "T0_none":
                    continue
                intervals = policy_intervals(
                    policy,
                    domain=domain,
                    source_tokens=len(source_tokens),
                    evidence_core_spans=evidence_core,
                    evidence_fact_spans=evidence_fact,
                    wrong_core_spans=wrong_core,
                ) if policy.mode != "none" else []
                positions = materialized_positions(
                    policy, intervals, source_tokens=len(source_tokens)
                )
                row, trajectory, heads = _condition(
                    model,
                    capture,
                    parent,
                    example,
                    policy,
                    intervals,
                    positions,
                    evidence_set,
                    distractor_set,
                    layer=PRIMARY_LAYER,
                    baseline_states=baseline_states,
                    baseline_pra=baseline_pra,
                    device=device,
                )
                if policy.name == "T0_none":
                    baseline_states = {
                        layer: state.clone() for layer, state in capture.states.items()
                    }
                    baseline_pra = {
                        layer: state.clone()
                        for layer, state in capture.pra_outputs.items()
                    }
                if not should_record:
                    continue
                common = {
                    "window": window_name(window),
                    "W": window_name(window),
                    "seed": seed,
                    "partition": partition,
                    "example_id": example.example_id,
                    "depth": example.depth,
                    "policy": policy.name,
                    "mode": policy.mode,
                    "radius": policy.radius,
                    "budget": policy.budget,
                    "allocation": policy.allocation,
                    "wrong_memory": policy.wrong_memory,
                    "region_groups": policy.region_groups,
                    "query_tokens": len(example.query_input_ids),
                    "evidence_region_count": len(evidence_core),
                    "evidence_source_tokens": len(evidence_set),
                    "evidence_span": max(end for _, end in evidence_core)
                    - min(start for start, _ in evidence_core),
                    "max_evidence_gap": example.evidence_gap + 5,
                    "fact_tokens": 5,
                    "parent_tokens": len(source_tokens),
                    "intervals": json.dumps(
                        [[interval.start, interval.end] for interval in union_intervals(intervals)]
                    ),
                }
                record = {
                    "kind": "main",
                    "key": list(key),
                    "row": {**common, **row},
                    "trajectory": [{**common, **stage} for stage in trajectory],
                    "heads": [{**common, **head} for head in heads],
                }
                _append_record(checkpoint_log, record)
                completed.add(key)
    except Exception:
        capture.close()
        raise

    for profile_layer in PROFILE_LAYERS:
        profile_model = (
            model
            if profile_layer == PRIMARY_LAYER
            else portability_model
            if profile_layer == PORTABILITY_LAYER
            else _build_model(source, profile_layer, device)
        )
        profile_capture = capture if profile_layer == PRIMARY_LAYER else StateCapture(profile_model)
        try:
            for example_index, example in enumerate(examples):
                partition = "validation" if example_index < len(examples) // 2 else "heldout"
                source_tokens, fact_spans, core_spans = source_and_fact_spans(example)
                domain = f"controlled-parent://{example.example_id}"
                parent = _parent_entry(
                    profile_model,
                    tokenizer,
                    example,
                    source_tokens,
                    domain,
                    device,
                )
                geometry = _evidence_geometry(example, fact_spans, core_spans)
                evidence_core, evidence_fact, wrong_core, evidence_set, distractor_set = geometry
                baseline_states = baseline_pra = None
                profile_policies = (
                    ToyPolicy("T0_none", "none"),
                    ToyPolicy("T1_radius_0", "logical_intervals", radius=0),
                    ToyPolicy("T3_radius_4", "logical_intervals", radius=4),
                    ToyPolicy("T7_whole_parent", "selected_chunks", whole_parent=True),
                    ToyPolicy("T9_wrong_exact", "logical_intervals", wrong_memory=True),
                )
                for policy in profile_policies:
                    key = (
                        "profile",
                        window_name(window),
                        seed,
                        example.example_id,
                        profile_layer,
                        policy.name,
                    )
                    should_record = key not in completed
                    if not should_record and policy.name != "T0_none":
                        continue
                    intervals = policy_intervals(
                        policy,
                        domain=domain,
                        source_tokens=len(source_tokens),
                        evidence_core_spans=evidence_core,
                        evidence_fact_spans=evidence_fact,
                        wrong_core_spans=wrong_core,
                    ) if policy.mode != "none" else []
                    positions = materialized_positions(
                        policy, intervals, source_tokens=len(source_tokens)
                    )
                    row, trajectory, _heads = _condition(
                        profile_model,
                        profile_capture,
                        parent,
                        example,
                        policy,
                        intervals,
                        positions,
                        evidence_set,
                        distractor_set,
                        layer=profile_layer,
                        baseline_states=baseline_states,
                        baseline_pra=baseline_pra,
                        device=device,
                    )
                    if policy.name == "T0_none":
                        baseline_states = {
                            layer: state.clone()
                            for layer, state in profile_capture.states.items()
                        }
                        baseline_pra = {
                            layer: state.clone()
                            for layer, state in profile_capture.pra_outputs.items()
                        }
                    if not should_record:
                        continue
                    common = {
                        "window": window_name(window),
                        "seed": seed,
                        "partition": partition,
                        "example_id": example.example_id,
                        "depth": example.depth,
                        "consumer_layer": profile_layer,
                        "policy": policy.name,
                    }
                    _append_record(
                        checkpoint_log,
                        {
                            "kind": "profile",
                            "key": list(key),
                            "row": {**common, **row},
                            "trajectory": [
                                {**common, **stage} for stage in trajectory
                            ],
                        },
                    )
                    completed.add(key)
        finally:
            if profile_layer not in {PRIMARY_LAYER, PORTABILITY_LAYER}:
                profile_capture.close()
                del profile_model
            elif profile_layer == PORTABILITY_LAYER:
                profile_capture.close()
    capture.close()
    del model, portability_model, source
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def _export(checkpoint_log: Path, output: Path) -> None:
    records = _checkpoint_records(checkpoint_log)
    main = [record["row"] for record in records if record["kind"] == "main"]
    trajectories = [
        row
        for record in records
        if record["kind"] == "main"
        for row in record.get("trajectory", [])
    ]
    attention = [
        row
        for record in records
        if record["kind"] == "main"
        for row in record.get("heads", [])
    ]
    portability = [
        record["row"] for record in records if record["kind"] == "portability"
    ]
    profile = [record["row"] for record in records if record["kind"] == "profile"]
    profile_trajectories = [
        row
        for record in records
        if record["kind"] == "profile"
        for row in record.get("trajectory", [])
    ]
    _write_csv(output / "toy_materialization_rows.csv", main)
    _write_csv(output / "toy_margin_trajectories.csv", trajectories)
    _write_csv(output / "toy_attention_decomposition.csv", attention)
    _write_csv(output / "toy_portability.csv", portability)
    _write_csv(output / "toy_consumer_layer_profile.csv", profile)
    _write_csv(output / "toy_consumer_layer_trajectories.csv", profile_trajectories)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--windows", default="16,32,64,128,global")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--examples", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--layers", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_log = args.output_dir / "toy_materialization_checkpoint.jsonl"
    existing = _checkpoint_records(checkpoint_log)
    completed = {tuple(record["key"]) for record in existing}
    tokenizer = ControlledTokenizer()
    examples = controlled_examples(
        tokenizer,
        count=args.examples,
        seed=100_004,
        depths=(1, 2, 3, 4),
        distractors=(4, 8),
        evidence_gaps=(0, 2, 6),
        lexical_overlaps=(0.0, 0.5, 1.0),
        relation_types=(4, 8, 15),
        branchings=(0, 1, 2),
    )
    config = {
        "schema_version": "1.0",
        "protocol": "oracle-parent toy native-K/V materialization",
        "windows": [window_name(value) for value in parse_windows(args.windows)],
        "seeds": [int(value) for value in args.seeds.split(",")],
        "examples": args.examples,
        "validation_examples": args.examples // 2,
        "heldout_examples": args.examples - args.examples // 2,
        "primary_consumer_layer": PRIMARY_LAYER,
        "portability_layer": PORTABILITY_LAYER,
        "consumer_profile_layers": list(PROFILE_LAYERS),
        "policies": [asdict(policy) for policy in policies()],
        "checkpoint_directory": str(args.checkpoint_dir),
        "selection": "single oracle parent; no competing routing identity",
        "weights_frozen": True,
        "device": args.device,
    }
    (args.output_dir / "toy_materialization_configs.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    for window in parse_windows(args.windows):
        for seed in [int(value) for value in args.seeds.split(",")]:
            checkpoint = (
                args.checkpoint_dir / f"{window_name(window)}_seed{seed}_best.pt"
            )
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            print(
                f"[toy] {window_name(window)} seed={seed} "
                f"checkpoint={digest[:12]}",
                flush=True,
            )
            _run_model(
                window=window,
                seed=seed,
                examples=examples,
                tokenizer=tokenizer,
                checkpoint=checkpoint,
                checkpoint_log=checkpoint_log,
                completed=completed,
                device=args.device,
                d_model=args.d_model,
                layers=args.layers,
            )
            _export(checkpoint_log, args.output_dir)
    print(f"wrote controlled materialization rows to {args.output_dir}")


if __name__ == "__main__":
    main()
