"""Optional mini-swe-agent Docker environment with Paper 8.5 provenance.

Select this class through mini-swe-agent's existing
``environment.environment_class`` configuration key. Rendered observation text
is unchanged; instrumentation is attached under the message ``extra`` mapping.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from minisweagent.environments.docker import DockerEnvironment, DockerEnvironmentConfig

from .observation_instrumentation import (
    build_bash_observation_metadata,
    stable_environment_fingerprint,
)
from .recordizer import extract_resource_ids


class InstrumentedDockerEnvironmentConfig(DockerEnvironmentConfig):
    instrumentation_output_root: str | None = None
    capture_workspace_checkpoints: bool = True
    visible_output_limit: int = 10_000


class InstrumentedDockerEnvironment(DockerEnvironment):
    """Capture pre/post versions and restorable pre-action repository state."""

    def __init__(self, **kwargs):
        super().__init__(config_class=InstrumentedDockerEnvironmentConfig, **kwargs)
        self._instrumentation_step = 0
        self._instrumentation_directory: Path | None = None
        if self.config.instrumentation_output_root:
            identity = (self.container_id or "unknown")[:12]
            self._instrumentation_directory = (
                Path(self.config.instrumentation_output_root) / identity
            )
            self._instrumentation_directory.mkdir(parents=True, exist_ok=True)
        self._environment_fingerprint = stable_environment_fingerprint({
            "image": self.config.image,
            "cwd": self.config.cwd,
            "env": self.config.env,
            "forward_env": self.config.forward_env,
            "interpreter": self.config.interpreter,
        })

    def serialize(self) -> dict:
        serialized = super().serialize()
        serialized["info"] = {
            **serialized.get("info", {}),
            "paper8_5_instrumentation": {
                "schema_version": 1,
                "environment_fingerprint": self._environment_fingerprint,
                "checkpoint_directory": (
                    str(self._instrumentation_directory)
                    if self._instrumentation_directory else None
                ),
                "executed_actions": self._instrumentation_step,
            },
        }
        return serialized

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        command = str(action.get("command", ""))
        effective_cwd = cwd or self.config.cwd
        step = self._instrumentation_step
        self._instrumentation_step += 1
        resources = extract_resource_ids(command, "")
        pre_state = self._capture_state(effective_cwd, resources)
        effective_cwd = str(pre_state.get("cwd") or effective_cwd)
        checkpoint = self._capture_checkpoint(step, command, effective_cwd, pre_state)
        output = super().execute(action, cwd=cwd, timeout=timeout)
        post_state = self._capture_state(effective_cwd, resources)
        metadata = build_bash_observation_metadata(
            command=command,
            cwd=effective_cwd,
            raw_output=str(output.get("output", "")),
            return_code=int(output.get("returncode", -1)),
            exception_info=str(output.get("exception_info", "")),
            pre_state=pre_state,
            post_state=post_state,
            environment_fingerprint=self._environment_fingerprint,
            visible_output_limit=self.config.visible_output_limit,
        )
        metadata.update({
            "paper8_5_execution_step": step,
            "paper8_5_pre_action_checkpoint": checkpoint,
        })
        output["extra"] = {**dict(output.get("extra") or {}), **metadata}
        self._write_receipt(step, command, pre_state, post_state, metadata)
        return output

    def _capture_state(self, cwd: str, resources: tuple[str, ...]) -> dict[str, Any]:
        encoded = base64.b64encode(json.dumps(list(resources)).encode()).decode()
        script = f'''python3 - <<'PY'
import base64, hashlib, json, os, subprocess
resources = json.loads(base64.b64decode("{encoded}"))
def run(*args):
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.returncode, p.stdout
def digest(data): return hashlib.sha256(data).hexdigest()
versions = {{}}
for resource in resources:
    path = os.path.realpath(resource)
    if os.path.isfile(path):
        with open(path, "rb") as handle: versions[resource] = digest(handle.read())
    elif not os.path.exists(path):
        versions[resource] = "missing"
head_rc, head = run("git", "rev-parse", "HEAD")
index_rc, index = run("git", "diff", "--cached", "--binary", "--full-index")
work_rc, work = run("git", "diff", "--binary", "--full-index")
names_rc, names = run("git", "ls-files", "--others", "--exclude-standard", "-z")
untracked = []
if names_rc == 0:
    for raw_name in names.split(b"\\0"):
        if not raw_name:
            continue
        name = os.fsdecode(raw_name)
        path = os.path.realpath(name)
        if os.path.isfile(path):
            with open(path, "rb") as handle:
                untracked.append((name, "file", digest(handle.read())))
        elif os.path.isdir(path):
            untracked.append((name, "directory", None))
        elif os.path.lexists(path):
            untracked.append((name, "other", None))
complete = all(rc == 0 for rc in (head_rc, index_rc, work_rc, names_rc))
manifest = json.dumps(untracked, sort_keys=True, separators=(",", ":")).encode()
workspace = digest(b"\\0".join((head, index, work, manifest))) if complete else None
print(json.dumps({{"complete": complete, "cwd": os.path.realpath(os.getcwd()),
 "head": head.decode(errors="replace").strip(),
 "workspace_version_fingerprint": workspace,
 "untracked_manifest": untracked,
 "resource_version_fingerprints": versions}}))
PY'''
        probe = DockerEnvironment.execute(self, {"command": script}, cwd=cwd)
        if probe.get("returncode") != 0:
            return {"complete": False, "error": probe.get("output", "")}
        try:
            return dict(json.loads(str(probe.get("output", ""))))
        except json.JSONDecodeError:
            return {"complete": False, "error": "state probe returned invalid JSON"}

    def _capture_checkpoint(
        self,
        step: int,
        command: str,
        cwd: str,
        pre_state: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not self.config.capture_workspace_checkpoints or self._instrumentation_directory is None:
            return None
        stem = self._instrumentation_directory / f"decision_{step:04d}"
        index_result = self._docker_bytes(
            cwd, "git diff --cached --binary --full-index"
        )
        worktree_result = self._docker_bytes(
            cwd, "git diff --binary --full-index"
        )
        archive_result = self._docker_bytes(
            cwd,
            "git ls-files --others --exclude-standard -z | "
            "tar --null --files-from=- -czf - 2>/dev/null",
        )
        index_patch = index_result["stdout"]
        worktree_patch = worktree_result["stdout"]
        archive = archive_result["stdout"]
        index_path = stem.with_suffix(".index.patch")
        worktree_path = stem.with_suffix(".worktree.patch")
        archive_path = stem.with_suffix(".untracked.tar.gz")
        index_path.write_bytes(index_patch)
        worktree_path.write_bytes(worktree_patch)
        archive_path.write_bytes(archive)
        receipt = {
            "schema_version": 1,
            "scope": "git_HEAD_plus_tracked_diff_plus_nonignored_untracked_files",
            "step": step,
            "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
            "head": pre_state.get("head"),
            "workspace_version_fingerprint": pre_state.get(
                "workspace_version_fingerprint"
            ),
            "index_patch": index_path.name,
            "index_patch_sha256": hashlib.sha256(index_patch).hexdigest(),
            "index_capture_complete": index_result["returncode"] == 0,
            "worktree_patch": worktree_path.name,
            "worktree_patch_sha256": hashlib.sha256(worktree_patch).hexdigest(),
            "worktree_capture_complete": worktree_result["returncode"] == 0,
            "untracked_archive": archive_path.name,
            "untracked_archive_sha256": hashlib.sha256(archive).hexdigest(),
            "untracked_capture_complete": archive_result["returncode"] == 0,
            "checkpoint_complete": (
                pre_state.get("complete") is True
                and index_result["returncode"] == 0
                and worktree_result["returncode"] == 0
                and archive_result["returncode"] == 0
            ),
        }
        receipt_path = stem.with_suffix(".json")
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        return {**receipt, "receipt": receipt_path.name}

    def _docker_bytes(self, cwd: str, command: str) -> dict[str, Any]:
        assert self.container_id
        cmd = [self.config.executable, "exec", "-w", cwd]
        for key, value in self.config.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([self.container_id, *self.config.interpreter, command])
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.config.timeout,
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr.decode(errors="replace"),
        }

    def _write_receipt(
        self,
        step: int,
        command: str,
        pre_state: dict[str, Any],
        post_state: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        if self._instrumentation_directory is None:
            return
        path = self._instrumentation_directory / f"execution_{step:04d}.json"
        path.write_text(json.dumps({
            "schema_version": 1,
            "step": step,
            "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
            "pre_state": pre_state,
            "post_state": post_state,
            "observation_metadata": metadata,
        }, indent=2) + "\n", encoding="utf-8")
