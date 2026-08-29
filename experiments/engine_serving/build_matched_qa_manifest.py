"""Build the frozen cross-engine E0/E2 natural-QA selection manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.engine_serving.matched_qa import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("docs/papers/shared/results/paper6_2_mlx"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "pra")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/papers/shared/results/matched_e0_e2_qa_manifest.json"),
    )
    args = parser.parse_args()
    manifest = build_manifest(args.result_dir, args.cache_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
