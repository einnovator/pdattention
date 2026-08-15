"""Trace how routed controlled evidence affects frozen-model answer formation.

The runner reuses the validation-selected v6 checkpoints.  It changes neither
training nor routing: forward hooks observe the residual stream, while matched
cache interventions isolate selected, oracle, irrelevant, and content-shuffled
memory under identical native-K/V budgets.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from pra_torch.controlled_local_sa import (
    SPECIAL_TOKENS,
    ControlledExample,
    ControlledTokenizer,
    controlled_examples,
)
from pra_torch.memory import PRACacheEntry
from pra_torch.model import TinyPRAModel, convert_sa_model_to_pra

from experiments.paper2_5_iterative_pra.run_controlled_local_sa import (
    DEFAULT_OUTPUT,
    SEEDS,
    model_config,
    parse_windows,
    window_name,
)
from experiments.paper2_5_iterative_pra.run_controlled_pra import (
    _pra_patterns,
    precompute_reference_entries,
)


def label_metrics(logits: torch.Tensor, answer_id: int) -> dict[str, float | int]:
    """Return calibrated 8-label diagnostics from one ``[V]`` readout."""
    label_start = len(SPECIAL_TOKENS)
    label_logits = logits[label_start : label_start + 8]
    label_index = int(answer_id) - label_start
    if not 0 <= label_index < 8:
        raise ValueError("Controlled answer ID must name one of the eight label entities.")
    correct_logit = label_logits[label_index]
    alternatives = label_logits.clone()
    alternatives[label_index] = -torch.inf
    max_wrong = alternatives.max()
    label_probabilities = label_logits.softmax(dim=-1)
    return {
        "correct_logit": float(correct_logit.detach().cpu()),
        "max_wrong_logit": float(max_wrong.detach().cpu()),
        "correct_margin": float((correct_logit - max_wrong).detach().cpu()),
        "correct_probability": float(label_probabilities[label_index].detach().cpu()),
        "prediction_entropy": float(
            (-(label_probabilities * label_probabilities.clamp_min(1e-12).log()).sum())
            .detach()
            .cpu()
        ),
        "brier_score": float(
            (
                (
                    label_probabilities
                    - F.one_hot(
                        torch.tensor(label_index, device=logits.device),
                        num_classes=8,
                    ).to(label_probabilities.dtype)
                )
                .square()
                .sum()
            )
            .detach()
            .cpu()
        ),
        "correct": int(int(label_logits.argmax()) == label_index),
        "full_vocabulary_correct": int(int(logits.argmax()) == int(answer_id)),
    }


def partition_memory_attention(
    selected_uris: Iterable[str],
    selected_lengths: Iterable[int],
    per_head_weights: Iterable[Iterable[float]],
    evidence_uris: set[str],
) -> dict[str, float]:
    """Split final-query shared-softmax mass into evidence/distractor/native parts."""
    uris = list(selected_uris)
    lengths = [int(length) for length in selected_lengths]
    weights = torch.tensor(list(list(head) for head in per_head_weights), dtype=torch.float64)
    if weights.numel() == 0:
        return {
            "evidence_attention_mass": 0.0,
            "distractor_attention_mass": 0.0,
            "native_attention_mass": 1.0,
            "attention_mass_sum": 1.0,
        }
    evidence_mask = torch.zeros(weights.shape[-1], dtype=torch.bool)
    cursor = 0
    for uri, length in zip(uris, lengths):
        if uri in evidence_uris:
            evidence_mask[cursor : cursor + length] = True
        cursor += length
    if cursor != weights.shape[-1]:
        raise ValueError("Selected-memory spans do not match captured attention width.")
    evidence = float(weights[:, evidence_mask].sum(dim=-1).mean())
    distractor = float(weights[:, ~evidence_mask].sum(dim=-1).mean())
    native = max(1.0 - evidence - distractor, 0.0)
    return {
        "evidence_attention_mass": evidence,
        "distractor_attention_mass": distractor,
        "native_attention_mass": native,
        "attention_mass_sum": evidence + distractor + native,
    }


def _shuffle_entry_payloads(entries: list[PRACacheEntry]) -> list[PRACacheEntry]:
    """Keep each URI/gist searchable but rotate the native value-bearing K/V payload."""
    if len(entries) < 2:
        return entries
    shuffled = []
    donors = entries[1:] + entries[:1]
    for entry, donor in zip(entries, donors):
        layer_memory = {}
        for layer_id, memory in entry.layer_memory.items():
            donor_memory = donor.layer_memory[layer_id]
            chunks = []
            for chunk, donor_chunk in zip(memory.chunks, donor_memory.chunks):
                chunks.append(
                    replace(
                        chunk,
                        token_kv=donor_chunk.token_kv,
                        metadata={**chunk.metadata, "payload_shuffled_from": donor.uri},
                    )
                )
            layer_memory[layer_id] = replace(memory, chunks=chunks)
        shuffled.append(replace(entry, layer_memory=layer_memory))
    return shuffled


def select_cache_entries(
    entries: list[PRACacheEntry],
    example: ControlledExample,
    evidence_condition: str,
) -> list[PRACacheEntry]:
    """Build a matched causal cache without exposing labels to normal routing."""
    evidence = set(example.target_reference_uris)
    if evidence_condition == "e0":
        return []
    if evidence_condition == "selected":
        return entries
    if evidence_condition in {"oracle", "wrong"}:
        # Identity forcing happens per consumer layer. Keeping the complete
        # cache here ensures the control changes selection, not availability.
        return entries
    if evidence_condition == "shuffle":
        return _shuffle_entry_payloads(entries)
    raise ValueError(f"Unknown evidence condition: {evidence_condition}")


def forced_reference_plan(
    example: ControlledExample,
    *,
    evidence_condition: str,
    layer_ids: tuple[int, ...],
    top_k: int,
) -> dict[int, tuple[str, ...]] | None:
    """Return a unique, exactly budgeted oracle or irrelevant-memory plan."""
    if evidence_condition not in {"oracle", "wrong"}:
        return None
    gold = list(example.target_reference_uris)
    distractors = [ref.uri for ref in example.references if not ref.is_evidence]
    if evidence_condition == "oracle":
        ordered = gold + distractors
    else:
        ordered = distractors
    required = len(layer_ids) * top_k
    if len(ordered) < required:
        raise ValueError("Controlled example lacks enough unique facts for a matched plan.")
    plan = {}
    cursor = 0
    for layer_id in layer_ids:
        plan[layer_id] = tuple(ordered[cursor : cursor + top_k])
        cursor += top_k
    return plan


def _put_entries(model: TinyPRAModel, entries: Iterable[PRACacheEntry]) -> None:
    model.clear_pra_cache()
    for entry in entries:
        model.pra_cache.put(entry)


class MechanisticCollector:
    """Collect intermediate readouts and PRA branch diagnostics via forward hooks."""

    def __init__(self, model: TinyPRAModel) -> None:
        self.model = model
        self.handles = []
        self.context: dict = {}
        self.intervention_layers: tuple[int, ...] = ()
        self.pre_states: dict[int, torch.Tensor] = {}
        self.stage_rows: list[dict] = []
        self.attention_rows: list[dict] = []
        self.value_rows: list[dict] = []
        self.residual_rows: list[dict] = []
        self.alignment_rows: list[dict] = []
        self.head_rows: list[dict] = []
        self.state_vectors: dict[tuple[str, int], torch.Tensor] = {}
        for block in model.blocks:
            layer_id = int(block.layer_id)
            self.handles.append(block.register_forward_pre_hook(self._pre_hook(layer_id)))
            self.handles.append(block.register_forward_hook(self._post_hook(layer_id)))
            attention = getattr(block, "attn", None) or getattr(block, "pra_attn", None)
            if attention is not None and hasattr(attention, "last_selected_chunks"):
                self.handles.append(attention.register_forward_hook(self._attention_hook(layer_id)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def reset(self, *, intervention_layers: tuple[int, ...] = (), **context) -> None:
        self.context = context
        self.intervention_layers = tuple(intervention_layers)
        self.pre_states = {}
        self.stage_rows = []
        self.attention_rows = []
        self.value_rows = []
        self.residual_rows = []
        self.alignment_rows = []
        self.head_rows = []
        self.state_vectors = {}

    def _readout(self, state: torch.Tensor) -> dict:
        logits = self.model.head(self.model.ln(state[:, -1, :]))[0]
        return label_metrics(logits, self.context["answer_id"])

    def _stage(self, stage_type: str, layer: int, state: torch.Tensor) -> None:
        intervention_index = sum(
            int(intervention_layer <= layer)
            for intervention_layer in self.intervention_layers
        )
        self.stage_rows.append(
            {
                **self.context,
                "stage_type": stage_type,
                "layer": layer,
                # Zero means no PRA intervention has occurred yet; later values
                # count completed interventions along the evolving-state trace.
                "intervention_index": intervention_index,
                **self._readout(state),
            }
        )
        self.state_vectors[(stage_type, layer)] = state[:, -1, :].detach().float().cpu()

    def _pre_hook(self, layer_id: int):
        def hook(_module, inputs):
            state = inputs[0]
            self.pre_states[layer_id] = state
            if layer_id == 0:
                self._stage("input", -1, state)
        return hook

    def _post_hook(self, layer_id: int):
        def hook(_module, _inputs, output):
            self._stage("after_layer", layer_id, output)
        return hook

    def _answer_alignment(self, pre: torch.Tensor, update: torch.Tensor) -> float:
        with torch.enable_grad():
            state = pre[:, -1, :].detach().requires_grad_(True)
            logits = self.model.head(self.model.ln(state))[0]
            label_start = len(SPECIAL_TOKENS)
            answer = int(self.context["answer_id"]) - label_start
            label_logits = logits[label_start : label_start + 8]
            alternatives = label_logits.clone()
            alternatives[answer] = -torch.inf
            margin = label_logits[answer] - alternatives.max()
            gradient = torch.autograd.grad(margin, state)[0]
        return float(F.cosine_similarity(update[:, -1, :].detach(), gradient, dim=-1).mean().cpu())

    def _attention_hook(self, layer_id: int):
        def hook(module, _inputs, output):
            pre = self.pre_states[layer_id]
            self._stage("after_pra", layer_id, pre + output)
            selected = list(module.last_selected_chunks[0])
            stats = module.last_memory_batching_stats
            weights = stats.final_token_memory_weights[0] if stats and stats.final_token_memory_weights else ()
            uris = [hit.reference_uri for hit in selected]
            lengths = [hit.selected_token_count for hit in selected]
            partition = partition_memory_attention(
                uris,
                lengths,
                weights,
                set(self.context["target_reference_uris"]),
            )
            diagnostics = module.last_diagnostics
            common = {
                **self.context,
                "layer": layer_id,
                "selected_reference_uris": json.dumps(uris),
                "selected_token_count": sum(lengths),
                **partition,
            }
            self.attention_rows.append(
                {
                    **common,
                    "attention_entropy": float(diagnostics.get("memory_attention_entropy", 0.0)),
                    "effective_support": math.exp(float(diagnostics.get("memory_attention_entropy", 0.0))),
                    "memory_attention_mass": float(diagnostics.get("final_token_memory_attention_mass", 0.0)),
                }
            )
            pre_norm = float(pre[:, -1, :].norm().detach().cpu())
            update_norm = float(output[:, -1, :].norm().detach().cpu())
            post = pre + output
            self.residual_rows.append(
                {
                    **common,
                    "pre_residual_norm": pre_norm,
                    "attention_update_norm": update_norm,
                    "attention_update_ratio": update_norm / max(pre_norm, 1e-12),
                    "post_attention_displacement": float(
                        1.0 - F.cosine_similarity(pre[:, -1, :], post[:, -1, :], dim=-1).mean().detach().cpu()
                    ),
                    "pra_output_divergence_ratio": float(diagnostics.get("pra_output_divergence_ratio", 0.0)),
                }
            )
            self.alignment_rows.append(
                {**common, "answer_direction_alignment": self._answer_alignment(pre, output)}
            )
            if weights:
                weight_tensor = torch.tensor(weights, dtype=torch.float32)
                evidence = set(self.context["target_reference_uris"])
                cursor = 0
                for head_index, head in enumerate(weight_tensor):
                    evidence_mass = distractor_mass = 0.0
                    cursor = 0
                    for uri, length in zip(uris, lengths):
                        mass = float(head[cursor : cursor + length].sum())
                        if uri in evidence:
                            evidence_mass += mass
                        else:
                            distractor_mass += mass
                        cursor += length
                    self.head_rows.append(
                        {
                            **self.context,
                            "layer": layer_id,
                            "head": head_index,
                            "evidence_attention_mass": evidence_mass,
                            "distractor_attention_mass": distractor_mass,
                            "native_attention_mass": max(1.0 - evidence_mass - distractor_mass, 0.0),
                        }
                    )
                values = torch.cat([hit.chunk.token_kv.v for hit in selected], dim=2) if selected else None
                if values is not None and values.shape[2] == weight_tensor.shape[1]:
                    values = values[0].detach().float().cpu()
                    evidence_mask = torch.zeros(values.shape[1], dtype=torch.bool)
                    cursor = 0
                    for uri, length in zip(uris, lengths):
                        if uri in evidence:
                            evidence_mask[cursor : cursor + length] = True
                        cursor += length
                    evidence_vector = (weight_tensor[:, :, None] * values * evidence_mask[None, :, None]).sum(dim=1)
                    distractor_vector = (weight_tensor[:, :, None] * values * (~evidence_mask)[None, :, None]).sum(dim=1)
                    self.value_rows.append(
                        {
                            **common,
                            "evidence_value_contribution_norm": float(evidence_vector.norm()),
                            "distractor_value_contribution_norm": float(distractor_vector.norm()),
                            "evidence_to_distractor_contribution_ratio": float(evidence_vector.norm() / distractor_vector.norm().clamp_min(1e-12)),
                        }
                    )
        return hook


def _condition_specs(n_layers: int) -> list[tuple[str, str, tuple[int, ...], int]]:
    patterns = _pra_patterns(n_layers)
    specs = []
    for policy in ("one_shot", "iterative_matched"):
        for evidence in ("e0", "selected", "oracle", "wrong", "shuffle"):
            specs.append((policy, evidence, patterns[policy], 4 if policy == "one_shot" else 1))
    for policy in ("spacing_1", "spacing_2", "spacing_4"):
        specs.append((policy, "selected", patterns[policy], 1))
    for layer in range(n_layers):
        specs.append((f"oracle_layer_{layer}", "oracle", (layer,), 4))
    return specs


def _build_model(
    source: TinyPRAModel,
    layer_ids: tuple[int, ...],
    top_k: int,
    device: str,
) -> TinyPRAModel:
    cfg = replace(
        source.cfg,
        model_variant="td_layered_pra",
        pra_layer_ids=layer_ids,
        top_k_references=top_k,
        top_k_chunks_per_reference=1,
        trigger_threshold=-1.0,
        max_materialized_memory_tokens=20,
        collect_attention_metrics=True,
        collect_per_head_metrics=True,
        collect_routing_metrics=True,
    )
    return convert_sa_model_to_pra(source, cfg).to(device).eval()


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--windows", default="16,32,64,128,global")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--examples", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--layers", type=int, default=6)
    args = parser.parse_args()
    output = args.output_dir / "mechanistic"
    tokenizer = ControlledTokenizer()
    all_rows: dict[str, list[dict]] = {
        "condition": [],
        "stage": [],
        "attention": [],
        "value": [],
        "residual": [],
        "alignment": [],
        "head": [],
    }
    for window in parse_windows(args.windows):
        for seed in [int(value) for value in args.seeds.split(",")]:
            checkpoint = args.output_dir / "checkpoints" / f"{window_name(window)}_seed{seed}.pt"
            cfg = model_config(tokenizer, window=window, device=args.device, d_model=args.d_model, n_layers=args.layers)
            source = TinyPRAModel(cfg).to(args.device).eval()
            source.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=False)["model"])
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
            entries_by_example = precompute_reference_entries(source, examples, tokenizer, args.device)
            for policy, evidence_condition, layers, top_k in _condition_specs(args.layers):
                model = _build_model(source, layers, top_k, args.device)
                collector = MechanisticCollector(model)
                try:
                    for example in examples:
                        entries = select_cache_entries(entries_by_example[example.example_id], example, evidence_condition)
                        _put_entries(model, entries)
                        context = {
                            "example_id": example.example_id,
                            "seed": seed,
                            "window": window_name(window),
                            "depth": example.depth,
                            "policy": policy,
                            "evidence_condition": evidence_condition,
                            "condition": f"{policy}_{evidence_condition}",
                            "answer_id": example.answer_id,
                            "target_reference_uris": example.target_reference_uris,
                            "evidence_span_tokens": example.evidence_distance,
                            "max_hop_distance_tokens": example.evidence_gap + 5,
                            "span_over_window": (
                                example.evidence_distance / window if window is not None else 0.0
                            ),
                            "hop_over_window": (
                                (example.evidence_gap + 5) / window if window is not None else 0.0
                            ),
                            "query_token_count": len(example.query_input_ids),
                            "query_pooling": "final query token",
                            "query_state_layer": "pre-consumer-layer final-token state",
                            "fact_token_count": 5,
                            "W": window_name(window),
                        }
                        collector.reset(intervention_layers=layers, **context)
                        ids = torch.tensor([example.query_input_ids], dtype=torch.long, device=args.device)
                        with torch.no_grad():
                            logits, trace = model.forward_progressive_pra(
                                ids,
                                prevent_reference_replay=True,
                                forced_reference_uris_by_layer=forced_reference_plan(
                                    example,
                                    evidence_condition=evidence_condition,
                                    layer_ids=layers,
                                    top_k=top_k,
                                ),
                            )
                        final = label_metrics(logits[0, -1], example.answer_id)
                        selected = list(dict.fromkeys(uri for row in trace for uri in row["selected_reference_uris"]))
                        gold = set(example.target_reference_uris)
                        materialized_states = sum(int(row["materialized_tokens"]) for row in trace)
                        if policy in {"one_shot", "iterative_matched"}:
                            expected_states = 0 if evidence_condition == "e0" else 20
                            if materialized_states != expected_states:
                                raise RuntimeError(
                                    f"{policy}/{evidence_condition} used {materialized_states} "
                                    f"layer-token K/V states; expected {expected_states}."
                                )
                        condition_row = {
                            **context,
                            **final,
                            "selected_reference_uris": json.dumps(selected),
                            "reference_recall": len(set(selected) & gold) / max(len(gold), 1),
                            "complete_path_recovery": int(gold <= set(selected)),
                            "layer_token_kv_states": materialized_states,
                        }
                        all_rows["condition"].append(condition_row)
                        for row in collector.stage_rows:
                            all_rows["stage"].append({**row, "final_correct": final["correct"], "final_margin": final["correct_margin"]})
                        all_rows["attention"].extend(collector.attention_rows)
                        all_rows["value"].extend(collector.value_rows)
                        all_rows["residual"].extend(collector.residual_rows)
                        all_rows["alignment"].extend(collector.alignment_rows)
                        all_rows["head"].extend(collector.head_rows)
                finally:
                    collector.close()
                print(f"completed {window_name(window)} seed={seed} {policy}/{evidence_condition}", flush=True)
    names = {
        "condition": "causal_memory_ablation.csv",
        "stage": "answer_margin_trajectory_rows.csv",
        "attention": "memory_attention_decomposition.csv",
        "value": "memory_value_contribution.csv",
        "residual": "residual_update_decomposition.csv",
        "alignment": "answer_direction_alignment.csv",
        "head": "head_memory_usage.csv",
    }
    for key, filename in names.items():
        _write_csv(output / filename, all_rows[key])
    _write_csv(
        output / "toy_query_geometry.csv",
        [
            {
                key: row[key]
                for key in (
                    "example_id", "seed", "window", "depth", "query_token_count",
                    "query_pooling", "query_state_layer", "fact_token_count", "W",
                    "evidence_span_tokens", "max_hop_distance_tokens",
                    "span_over_window", "hop_over_window",
                )
            }
            for row in all_rows["condition"]
            if row["condition"] == "one_shot_e0"
        ],
    )
    print(f"wrote mechanistic captures to {output}")


if __name__ == "__main__":
    main()
