"""Run Paper 6.5 M6 co-located native-Q/K resource discovery.

The frozen Qwen model encodes structured tool definitions once. Online search
compares lexical/index controls, deterministic external semantic hashing,
input-embedding means, native mean K, full token QK, and zero-shot Paper 2.8
low-rank QK indexes. Raw Q/K never appears in the persisted result rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.agent_workflows import realistic_tool_catalog, workflow_tasks
from experiments.paper2_8_qk_compression.run_query_conditioned_study import _query_feature
from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper6_5_tools.run_m2_m4_pretrained import SEEDS, _prompt_variant, _write_csv
from pra_hf.agent_resources import (
    DiscoveryRequest,
    PersistentResourceIndex,
    hashed_semantic_vector,
)
from pra_hf.native_resource_discovery import native_mean_k_scores, native_token_qk_scores
from pra_hf.qk_compression import (
    QueryConditionedLandmarkSelector,
    kmeans_centroids,
    low_rank_response_scores,
)


MODES = (
    "token",
    "index",
    "external_signed_hash",
    "input_embedding_mean",
    "native_mean_k",
    "native_token_qk",
    "paper2_8_rank16_ensemble",
    "paper2_8_rank8_centroids",
    "lexical_native_hybrid",
)
ROUTING_LAYER = 27


def _structured_definition(resource) -> str:
    """Encode structural fields with explicit boundaries rather than prose flattening."""

    return "\n".join((
        f"NAME: {resource.name}",
        f"DESCRIPTION: {resource.description}",
        f"PARAMETERS: {resource.content}",
        f"RETURNS: {json.dumps(resource.metadata.get('returns', {}), sort_keys=True)}",
        f"SIDE_EFFECT: {resource.side_effect_class.value}",
        f"TAGS: {json.dumps(sorted(resource.tags))}",
    ))


class NativeFeatureEncoder:
    """Extract pre-RoPE Q/K and input-embedding controls at one frozen layer."""

    def __init__(self, model_id: str, revision: str, device: torch.device, layer: int) -> None:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.device = device
        self.layer = layer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=True,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
        ).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.attention = self.model.model.layers[layer].self_attn

    def encode(self, text: str) -> dict[str, torch.Tensor]:
        tokens = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        with torch.inference_mode():
            outputs = self.model.model(
                **tokens,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states[self.layer]
            q = self.attention.q_proj(hidden).view(
                hidden.shape[0], hidden.shape[1], self.attention.config.num_attention_heads, self.attention.head_dim
            )
            k = self.attention.k_proj(hidden).view(
                hidden.shape[0], hidden.shape[1], self.attention.config.num_key_value_heads, self.attention.head_dim
            )
            if hasattr(self.attention, "q_norm"):
                q = self.attention.q_norm(q)
            if hasattr(self.attention, "k_norm"):
                k = self.attention.k_norm(k)
            embeddings = self.model.get_input_embeddings()(tokens.input_ids)
        span = min(8, q.shape[1])
        return {
            "q": q[0, -span:].float().mean(dim=0).cpu(),
            "k": k[0].float().cpu(),
            "embedding": embeddings[0].float().mean(dim=0).cpu(),
            "tokens": torch.tensor(int(tokens.input_ids.shape[1])),
        }


def _pad_keys(features: Sequence[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    max_tokens = max(int(row["k"].shape[0]) for row in features)
    sample = features[0]["k"]
    keys = torch.zeros((len(features), max_tokens, *sample.shape[1:]), dtype=torch.float32)
    mask = torch.zeros((len(features), max_tokens), dtype=torch.bool)
    for index, row in enumerate(features):
        count = row["k"].shape[0]
        keys[index, :count] = row["k"]
        mask[index, :count] = True
    return keys, mask


def _load_selectors(checkpoint_dir: Path, rank: int, device: torch.device):
    values = []
    for seed in SEEDS:
        checkpoint = torch.load(
            checkpoint_dir / f"direct_lowrank_r{rank}_seed{seed}.pt",
            map_location="cpu",
            weights_only=False,
        )
        selector = QueryConditionedLandmarkSelector(
            2048,
            feature_width=1024,
            rank=rank,
            use_salience=False,
            use_interaction=True,
        ).to(device).eval()
        selector.load_state_dict(checkpoint["state_dict"])
        values.append((selector, checkpoint))
    return tuple(values)


def _unit(values: torch.Tensor) -> torch.Tensor:
    low, high = values.min(), values.max()
    return (values - low) / (high - low).clamp_min(1e-12)


def _lowrank_scores(query, keys, mask, selectors, *, centroids: bool, device):
    query_feature = _query_feature(query).to(device)
    flat_keys = keys.flatten(2).to(device)
    mask = mask.to(device)
    rows = []
    for selector, checkpoint in selectors:
        projected = selector.feature_projection(flat_keys / float(checkpoint["native_key_rms_scale"]))
        projected_mask = mask
        if centroids:
            projected, projected_mask = kmeans_centroids(projected, projected_mask, 8)
        projected_query = selector.query_projection(query_feature)
        score = low_rank_response_scores(
            projected_query,
            projected,
            projected_mask,
            function="top_r_mean",
            top_r=4,
        )[0]
        rows.append(_unit(score.float()))
    return torch.stack(rows).mean(dim=0).cpu()


def _channel_scores(
    query_text: str,
    query_feature: dict,
    resources,
    resource_features,
    keys,
    mask,
    index,
    selectors16,
    selectors8,
    device,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    request = DiscoveryRequest(query=query_text, tenant_id="paper6_5", top_k=len(resources))
    lexical_rows = index.score(request, channels=("token", "index"))
    lexical_by_uri = {row.uri: row for row in lexical_rows}
    token = torch.tensor([lexical_by_uri[resource.uri].token for resource in resources])
    indexed = torch.tensor([lexical_by_uri[resource.uri].index for resource in resources])
    query_hash = torch.tensor(hashed_semantic_vector(query_text), dtype=torch.float32)
    hashes = torch.tensor(
        [hashed_semantic_vector(resource.search_text) for resource in resources], dtype=torch.float32
    )
    external = F.cosine_similarity(hashes, query_hash.unsqueeze(0), dim=-1).clamp_min(0.0)
    embeddings = torch.stack([row["embedding"] for row in resource_features])
    input_mean = F.cosine_similarity(embeddings, query_feature["embedding"].unsqueeze(0), dim=-1)
    q = query_feature["q"].to(device)
    gpu_keys = keys.to(device)
    gpu_mask = mask.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    native_mean = native_mean_k_scores(q, gpu_keys, gpu_mask).float()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    mean_seconds = time.perf_counter() - started
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    native_tokens = native_token_qk_scores(q, gpu_keys, gpu_mask, top_r=4).float()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    token_seconds = time.perf_counter() - started
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    low16 = _lowrank_scores(query_feature["q"], keys, mask, selectors16, centroids=False, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    low16_seconds = time.perf_counter() - started
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    low8 = _lowrank_scores(query_feature["q"], keys, mask, selectors8, centroids=True, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    low8_seconds = time.perf_counter() - started
    native_mean = native_mean.cpu()
    native_tokens = native_tokens.cpu()
    hybrid = 0.45 * _unit(torch.maximum(token, indexed)) + 0.55 * _unit(low16)
    scores = {
        "token": token,
        "index": indexed,
        "external_signed_hash": external,
        "input_embedding_mean": input_mean,
        "native_mean_k": native_mean,
        "native_token_qk": native_tokens,
        "paper2_8_rank16_ensemble": low16,
        "paper2_8_rank8_centroids": low8,
        "lexical_native_hybrid": hybrid,
    }
    timings = {
        "native_mean_k": mean_seconds,
        "native_token_qk": token_seconds,
        "paper2_8_rank16_ensemble": low16_seconds,
        "paper2_8_rank8_centroids": low8_seconds,
    }
    return scores, timings


def _metric_row(task, seed, mode, scores, resources, seconds, index_bytes):
    order = torch.argsort(scores, descending=True, stable=True).tolist()
    names = [resources[index].name for index in order]
    targets = set(task.required_tools)
    budget = 1 if len(task.steps) == 1 else len(task.steps)
    selected = names[:budget]
    ranks = [names.index(name) + 1 for name in targets]
    successors = targets - {task.required_tools[0]}
    return {
        "seed": seed,
        "task_id": task.task_id,
        "family": task.family,
        "plan_horizon": len(task.steps),
        "mode": mode,
        "budget": budget,
        "top1_correct": names[0] in targets,
        "mrr": 1.0 / min(ranks),
        "required_recall_at_budget": len(set(selected) & targets) / len(targets),
        "successor_recall_at_budget": (
            len(set(selected) & successors) / len(successors) if successors else float(names[0] in targets)
        ),
        "all_required_recovered": targets <= set(selected),
        "selected_tools": " ".join(selected),
        "first_target_rank": min(ranks),
        "routing_seconds": seconds,
        "index_bytes": index_bytes,
    }


def run(args) -> dict:
    device = torch.device(args.device)
    encoder = NativeFeatureEncoder(args.model_id, args.revision, device, args.layer)
    resources = realistic_tool_catalog()
    resource_features = []
    for index, resource in enumerate(resources, start=1):
        resource_features.append(encoder.encode(_structured_definition(resource)))
        print(f"[M6 encode resource {index}/{len(resources)}] {resource.name}", flush=True)
    keys, mask = _pad_keys(resource_features)
    index = PersistentResourceIndex(resources)
    selectors16 = _load_selectors(args.checkpoint_dir, 16, device)
    selectors8 = _load_selectors(args.checkpoint_dir, 8, device)
    token_count = int(mask.sum())
    kv_width = int(keys.shape[2] * keys.shape[3])
    index_bytes = {
        "token": index.estimated_bytes,
        "index": index.estimated_bytes,
        "external_signed_hash": len(resources) * 128 * 4,
        "input_embedding_mean": len(resources) * encoder.model.config.hidden_size * 2,
        "native_mean_k": len(resources) * kv_width * 2,
        "native_token_qk": token_count * kv_width * 2,
        "paper2_8_rank16_ensemble": token_count * 16 * 2,
        "paper2_8_rank8_centroids": len(resources) * 8 * 8 * 2,
        "lexical_native_hybrid": index.estimated_bytes + token_count * 16 * 2,
    }
    tasks = list(workflow_tasks())
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]
    rows = []
    for seed in args.seeds:
        for task in tasks:
            query = _prompt_variant(task.query, seed)
            query_feature = encoder.encode(query)
            scores_by_mode, timings = _channel_scores(
                query,
                query_feature,
                resources,
                resource_features,
                keys,
                mask,
                index,
                selectors16,
                selectors8,
                device,
            )
            for mode, scores in scores_by_mode.items():
                rows.append(_metric_row(task, seed, mode, scores, resources, timings.get(mode, 0.0), index_bytes[mode]))
            print(f"[M6 seed={seed} task={task.task_id}]", flush=True)
    _write_csv(args.output_dir / "m6_rows.csv", rows)
    fingerprint = hashlib.sha256(
        json.dumps([resource.fingerprint_payload() for resource in resources], sort_keys=True).encode()
    ).hexdigest()
    manifest = {
        "schema_version": "1.0",
        "model_id": args.model_id,
        "model_revision": args.revision,
        "model_frozen": True,
        "routing_layer": args.layer,
        "representation": "pre_rope_native_q_and_k",
        "deployment_executed": "co_located",
        "deployment_modes_specified": ["co_located", "shared_memory", "model_server", "replicated_query"],
        "deployment_interfaces_implemented": ["co_located", "shared_memory_projected_query", "model_server_identity_reply"],
        "raw_native_state_persisted": False,
        "resource_index_fingerprint": fingerprint,
        "resource_count": len(resources),
        "resource_tokens": token_count,
        "modes": list(MODES),
        "index_bytes": index_bytes,
        "paper2_8_projection_provenance": "five HotpotQA/QASPER validation-trained checkpoints; zero-shot on tool definitions",
        "tool_supervision_for_low_rank_projection": False,
        "rows": len(rows),
        "seeds": list(args.seeds),
        "runtime": runtime_metadata(),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--layer", type=int, default=ROUTING_LAYER)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "docs/papers/shared/results/paper2_8_qk_compression/low_rank_frontier/checkpoints")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/papers/shared/results/paper6_5_tools/m6_native")
    args = parser.parse_args()
    args.seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
