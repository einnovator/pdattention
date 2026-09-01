"""Run a bounded model-backed smoke against a local or Hub PRA bundle."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from pra_hf.bundle import PRAModelBundle
from pra_hf.model import PRAForCausalLM


def run(
    bundle_source: str,
    *,
    bundle_revision: str | None,
    profile: str,
    device: str,
) -> dict:
    started = time.perf_counter()
    bundle = PRAModelBundle.from_pretrained(bundle_source, revision=bundle_revision)
    components = bundle.selected_learned_adapters(profile)
    routing = next(
        (
            path
            for name, path in components.items()
            if bundle.learned_adapters[name].get("type") in {"routing", "router"}
        ),
        None,
    )
    model_kwargs = {"revision": bundle.base_model["revision"]}
    if device == "cuda":
        model_kwargs.update(torch_dtype=torch.float16, device_map="cuda")
    model = PRAForCausalLM.from_pretrained(
        str(bundle.base_model["id"]),
        routing_adapter=routing,
        **model_kwargs,
    )
    if device != "cuda":
        model.model.to(device)
    reference = model.add_reference(
        "pra://bundle-smoke/evidence",
        text="Progressive Retrieval Attention uses query-addressed external context.",
    )
    route = model.route("What uses query-addressed external context?")
    generated = model.generate(
        "Answer with one word: two plus two is",
        max_new_tokens=2,
        do_sample=False,
    )
    return {
        "status": "PASS",
        "bundle": bundle_source,
        "bundle_revision": bundle.resolved_revision,
        "base_model": bundle.base_model["id"],
        "base_revision": bundle.base_model["revision"],
        "device": str(model.device),
        "profile": profile,
        "routing_adapter": str(routing) if routing else None,
        "adapter_parameters": (
            sum(parameter.numel() for parameter in model.router.parameters())
            if model.router is not None else 0
        ),
        "reference_tokens": reference.tokens,
        "selected_regions": len(route.selected),
        "generation": generated,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle")
    parser.add_argument("--bundle-revision")
    parser.add_argument("--profile", default="balanced")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(
        args.bundle,
        bundle_revision=args.bundle_revision,
        profile=args.profile,
        device=args.device,
    )
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
