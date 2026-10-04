"""Dependency-free source for the generic transactional replacement shim."""

from __future__ import annotations

import hashlib


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


def main():
    if len(sys.argv) != 4:
        print("usage: pra_replace PATH EXACT_OLD_TEXT EXACT_NEW_TEXT", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    if path.is_symlink() or not path.is_file():
        print("PRA_REPLACE_REJECTED path must be a regular non-symlink file", file=sys.stderr)
        return 2
    old = decode(sys.argv[2]).encode("utf-8")
    new = decode(sys.argv[3]).encode("utf-8")
    data = path.read_bytes()
    occurrences = data.count(old)
    if not old or occurrences != 1:
        print(
            f"PRA_REPLACE_REJECTED occurrences={occurrences}; file unchanged",
            file=sys.stderr,
        )
        return 2
    updated = data.replace(old, new, 1)
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


__all__ = ["TRANSACTIONAL_REPLACE_TOOL", "TRANSACTIONAL_REPLACE_TOOL_SHA256"]
