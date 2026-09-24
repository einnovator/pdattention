"""Launch ``mlx_lm.server`` with explicit process-wide memory limits.

This wrapper is an operational control for memory-constrained Apple Silicon
hosts.  It does not implement PRA or change request semantics.  Arguments not
consumed by the wrapper are forwarded unchanged to ``mlx_lm.server``.
"""

from __future__ import annotations

import argparse
import sys


def _bytes_from_gib(value: float) -> int:
    if value < 0:
        raise argparse.ArgumentTypeError("memory limits must be non-negative")
    return int(value * 1024**3)


def _server_args(*, disable_thinking: bool, forwarded: list[str]) -> list[str]:
    server_args = list(forwarded)
    if disable_thinking:
        if "--chat-template-args" in server_args:
            raise ValueError(
                "--mlx-disable-thinking cannot be combined with an explicit "
                "--chat-template-args value"
            )
        server_args.extend(("--chat-template-args", '{"enable_thinking": false}'))
    return server_args


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mlx-cache-limit-gib", type=float, default=0.0)
    parser.add_argument("--mlx-wired-limit-gib", type=float, default=10.0)
    parser.add_argument(
        "--mlx-disable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Forward an explicit enable_thinking=false chat-template argument. "
            "This is a serving-profile choice and is recorded in the launch "
            "command; it does not modify model weights or PRA selection."
        ),
    )
    args, forwarded = parser.parse_known_args()
    try:
        server_args = _server_args(
            disable_thinking=args.mlx_disable_thinking,
            forwarded=forwarded,
        )
    except ValueError as error:
        parser.error(str(error))

    import mlx.core as mx

    mx.set_cache_limit(_bytes_from_gib(args.mlx_cache_limit_gib))
    mx.set_wired_limit(_bytes_from_gib(args.mlx_wired_limit_gib))

    # mlx_lm.server.main() owns its argument parser and reads sys.argv.
    sys.argv = ["mlx_lm.server", *server_args]
    from mlx_lm.server import main as server_main

    server_main()


if __name__ == "__main__":
    main()
