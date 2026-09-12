"""Download a pinned Hugging Face snapshot with explicit bounded concurrency."""

from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--allow-pattern", action="append", required=True)
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()
    if args.max_workers <= 0:
        raise ValueError("max-workers must be positive")
    path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        allow_patterns=args.allow_pattern,
        max_workers=args.max_workers,
    )
    print(path)


if __name__ == "__main__":
    main()
