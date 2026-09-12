"""Restore and verify Paper 8.5 repository-state checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(repository: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    result = subprocess.run(
        args,
        cwd=repository,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({' '.join(args)}): "
            f"{result.stderr.decode(errors='replace')}"
        )
    return result.stdout


def repository_state(repository: str | Path) -> dict[str, Any]:
    """Return the same content-addressed state captured by the harness."""

    root = Path(repository).resolve()
    head = _run(root, "git", "rev-parse", "HEAD")
    index = _run(root, "git", "diff", "--cached", "--binary", "--full-index")
    worktree = _run(root, "git", "diff", "--binary", "--full-index")
    names = _run(root, "git", "ls-files", "--others", "--exclude-standard", "-z")
    manifest: list[tuple[str, str, str | None]] = []
    for raw_name in names.split(b"\0"):
        if not raw_name:
            continue
        name = raw_name.decode(errors="surrogateescape")
        path = root / name
        if path.is_file():
            manifest.append((name, "file", _sha256(path)))
        elif path.is_dir():
            manifest.append((name, "directory", None))
        elif path.exists() or path.is_symlink():
            manifest.append((name, "other", None))
    encoded_manifest = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode()
    fingerprint = hashlib.sha256(
        b"\0".join((head, index, worktree, encoded_manifest))
    ).hexdigest()
    return {
        "head": head.decode(errors="replace").strip(),
        "workspace_version_fingerprint": fingerprint,
        "untracked_manifest": manifest,
    }


def _safe_archive_members(archive: tarfile.TarFile, root: Path) -> None:
    for member in archive.getmembers():
        destination = (root / member.name).resolve()
        if root != destination and root not in destination.parents:
            raise ValueError(f"unsafe path in checkpoint archive: {member.name}")


def restore_repository_checkpoint(
    receipt_path: str | Path,
    repository: str | Path,
) -> dict[str, Any]:
    """Restore a checkpoint into a clean disposable repository clone."""

    receipt_file = Path(receipt_path).resolve()
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    if receipt.get("checkpoint_complete") is not True:
        raise ValueError("checkpoint receipt is incomplete")
    root = Path(repository).resolve()
    if not (root / ".git").exists():
        raise ValueError("restore target must be a Git worktree root")
    if _run(root, "git", "status", "--porcelain", "-z"):
        raise ValueError("restore target must be clean")

    artifacts = (
        ("index_patch", "index_patch_sha256"),
        ("worktree_patch", "worktree_patch_sha256"),
        ("untracked_archive", "untracked_archive_sha256"),
    )
    resolved: dict[str, Path] = {}
    for field, digest_field in artifacts:
        path = Path(receipt[field])
        if not path.is_absolute():
            path = receipt_file.parent / path
        path = path.resolve()
        if _sha256(path) != receipt[digest_field]:
            raise ValueError(f"checkpoint digest mismatch: {field}")
        resolved[field] = path

    _run(root, "git", "checkout", "--detach", str(receipt["head"]))
    index_patch = resolved["index_patch"].read_bytes()
    if index_patch:
        _run(root, "git", "apply", "--binary", "--index", "-", input_bytes=index_patch)
    worktree_patch = resolved["worktree_patch"].read_bytes()
    if worktree_patch:
        _run(root, "git", "apply", "--binary", "-", input_bytes=worktree_patch)
    with tarfile.open(resolved["untracked_archive"], "r:gz") as archive:
        _safe_archive_members(archive, root)
        archive.extractall(root)

    state = repository_state(root)
    if state["workspace_version_fingerprint"] != receipt["workspace_version_fingerprint"]:
        raise RuntimeError("restored workspace fingerprint does not match receipt")
    return state
