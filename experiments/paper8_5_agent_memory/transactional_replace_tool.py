"""Dependency-free source for the generic transactional replacement shim."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shlex


def transactional_edit_policy_rejection(command: str) -> str | None:
    """Classify shell edits that bypass the fail-closed replacement tool.

    This is deliberately narrow: read-only ``sed -n`` remains available, as
    do tests, package installation, patch inspection, and ordinary shell
    commands. The control targets the two failure modes observed in autonomous
    runs: unavailable interactive editors and unguarded in-place substitution.
    """

    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return None
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token in {"&&", "||", ";", "|"}:
            segments.append([])
        else:
            segments[-1].append(token)
    for segment in segments:
        while segment and "=" in segment[0] and not segment[0].startswith(("-", "/")):
            segment = segment[1:]
        while segment and segment[0] in {"command", "env", "sudo"}:
            segment = segment[1:]
        if not segment:
            continue
        executable = Path(segment[0]).name
        if executable in {"nano", "vi", "vim", "nvim", "emacs"}:
            return "interactive_editor"
        if executable == "sed" and any(
            token == "-i" or token.startswith("-i") or token.startswith("--in-place")
            for token in segment[1:]
        ):
            return "unguarded_in_place_substitution"
        if executable == "perl" and any(
            token.startswith("-") and "i" in token[1:]
            for token in segment[1:]
        ):
            return "unguarded_in_place_substitution"
    return None


TRANSACTIONAL_REPLACE_TOOL = r'''#!/usr/bin/env python3
import hashlib
import os
from pathlib import Path
import stat
import sys
import tempfile


def decode(value):
    mapping = {"n": "\n", "r": "\r", "t": "\t", "\\": "\\"}
    result = []
    index = 0
    while index < len(value):
        if value[index] == "\\" and index + 1 < len(value):
            escaped = value[index + 1]
            if escaped in mapping:
                result.append(mapping[escaped])
                index += 2
                continue
        result.append(value[index])
        index += 1
    return "".join(result)


def candidate_hints(data, old):
    stripped = old.strip()
    if not stripped or b"\n" in stripped:
        return "none"
    hints = []
    for line_number, line in enumerate(data.splitlines(), 1):
        if old in line or line.strip() == stripped:
            leading = len(line) - len(line.lstrip(b" \t"))
            hints.append(f"line={line_number}:leading_bytes={leading}")
            if len(hints) == 4:
                break
    return ",".join(hints) if hints else "none"


def main():
    if len(sys.argv) not in (4, 5):
        print(
            "usage: pra_replace PATH EXACT_OLD_TEXT EXACT_NEW_TEXT [--line=N]",
            file=sys.stderr,
        )
        return 2
    line_number = None
    if len(sys.argv) == 5:
        selector = sys.argv[4]
        if not selector.startswith("--line=") or not selector[7:].isdigit():
            print(
                "PRA_REPLACE_REJECTED optional selector must be --line=N; "
                "file unchanged",
                file=sys.stderr,
            )
            return 2
        line_number = int(selector[7:])
        if line_number < 1:
            print(
                "PRA_REPLACE_REJECTED --line=N is one-based; file unchanged",
                file=sys.stderr,
            )
            return 2
    path = Path(sys.argv[1])
    if path.is_symlink() or not path.is_file():
        print("PRA_REPLACE_REJECTED path must be a regular non-symlink file", file=sys.stderr)
        return 2
    old = decode(sys.argv[2]).encode("utf-8")
    new = decode(sys.argv[3]).encode("utf-8")
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    if line_number is None:
        selected = data
    elif line_number > len(lines):
        print(
            f"PRA_REPLACE_REJECTED line={line_number} out_of_range; file unchanged",
            file=sys.stderr,
        )
        return 2
    else:
        selected = lines[line_number - 1]
    occurrences = selected.count(old)
    if not old or occurrences != 1:
        print(
            f"PRA_REPLACE_REJECTED occurrences={occurrences} "
            f"candidates={candidate_hints(data, old)}; file unchanged. "
            "For duplicate single-line matches inspect the intended candidate "
            "and append --line=N, or include bounded adjacent context in "
            "EXACT_OLD_TEXT.",
            file=sys.stderr,
        )
        return 2
    if line_number is None:
        updated = data.replace(old, new, 1)
    else:
        match_offset = selected.index(old)
        unchanged_prefix = selected[:match_offset]
        effective_new = new
        if (
            b"\n" in new
            and unchanged_prefix
            and not unchanged_prefix.strip(b" \t")
        ):
            effective_new = new.replace(b"\n", b"\n" + unchanged_prefix)
        lines[line_number - 1] = selected.replace(old, effective_new, 1)
        updated = b"".join(lines)
    original_mode = stat.S_IMODE(path.stat().st_mode)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".pra-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(updated)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, original_mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(
        "PRA_REPLACE_OK "
        f"path={path} old_sha256={hashlib.sha256(data).hexdigest()} "
        f"new_sha256={hashlib.sha256(updated).hexdigest()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
TRANSACTIONAL_REPLACE_TOOL_SHA256 = hashlib.sha256(
    TRANSACTIONAL_REPLACE_TOOL.encode("utf-8")
).hexdigest()


__all__ = [
    "TRANSACTIONAL_REPLACE_TOOL",
    "TRANSACTIONAL_REPLACE_TOOL_SHA256",
    "transactional_edit_policy_rejection",
]
