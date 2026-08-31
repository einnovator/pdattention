"""Create ordinary AirLLM safetensor shards on macOS for a CUDA host.

AirLLM selects its MLX persister automatically on Darwin. This utility is an
explicit portability tool: it selects the ordinary safetensor persister so a
model already cached on a Mac can be prepared for a separate CUDA machine.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import airllm.persist.model_persister as persister_module
from airllm.persist.safetensor_model_persister import SafetensorModelPersister
from airllm.utils import split_and_save_layers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    persister_module.model_persister = SafetensorModelPersister()
    result = split_and_save_layers(
        args.checkpoint,
        layer_shards_saving_path=args.output,
        layer_names={
            "embed": "model.embed_tokens",
            "layer_prefix": "model.layers",
            "norm": "model.norm",
            "lm_head": "lm_head",
        },
    )
    print(result)


if __name__ == "__main__":
    main()
