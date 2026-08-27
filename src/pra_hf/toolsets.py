"""Composable tool bundles for the product-facing PRA agent SDK."""

from __future__ import annotations

import dataclasses
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping

from .agent_execution import SafeToolExecutor
from .agent_resources import SideEffectClass
from .tool_records import ToolRecord, tool_record_from_callable


def _json_result(value: object) -> Mapping[str, object]:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return {"items": list(value)}
    if isinstance(value, bytes):
        return {"text": value.decode("utf-8", errors="replace")}
    return {"result": value}


@dataclass(frozen=True)
class Tool:
    """One callable plus execution policy kept beside its generated schema."""

    function: Callable[..., object]
    side_effect: SideEffectClass | str = SideEffectClass.NONE
    namespace: str = "default"
    tenant_id: str = "default"
    version: str = "v1"
    aliases: tuple[str, ...] = ()
    tags: frozenset[str] = frozenset()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise TypeError("Tool.function must be callable.")
        object.__setattr__(self, "side_effect", SideEffectClass(self.side_effect))
        object.__setattr__(self, "aliases", tuple(dict.fromkeys(self.aliases)))
        object.__setattr__(self, "tags", frozenset(self.tags))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def record(self) -> ToolRecord:
        return tool_record_from_callable(
            self.function,
            namespace=self.namespace,
            tenant_id=self.tenant_id,
            version=self.version,
            aliases=self.aliases,
            manual_tags=self.tags,
            metadata={
                **self.metadata,
                "side_effect_class": self.side_effect.value,
                "toolset_managed": True,
            },
        )

    def invoke(self, arguments: Mapping[str, object]) -> Mapping[str, object]:
        value = self.function(**dict(arguments))
        if hasattr(value, "__await__"):
            raise TypeError("Async tools require an asynchronous executor.")
        return _json_result(value)


class Toolset:
    """Named collection of tools with one safe executor and capability view."""

    def __init__(self, tools: Iterable[Tool], *, name: str = "custom") -> None:
        self.name = name
        self.tools = tuple(tools)
        self.records = tuple(tool.record for tool in self.tools)
        self.resources = tuple(record.to_agent_resource() for record in self.records)
        names = [resource.name for resource in self.resources]
        uris = [resource.uri for resource in self.resources]
        if len(names) != len(set(names)) or len(uris) != len(set(uris)):
            raise ValueError("Toolset tool names and URIs must be unique.")
        self._by_uri = dict(zip(uris, self.tools))

    @classmethod
    def from_callables(
        cls,
        functions: Iterable[Callable[..., object]],
        *,
        name: str = "custom",
        namespace: str = "default",
        tenant_id: str = "default",
    ) -> "Toolset":
        return cls(
            (Tool(function, namespace=namespace, tenant_id=tenant_id) for function in functions),
            name=name,
        )

    @classmethod
    def merge(cls, *toolsets: "Toolset", name: str = "merged") -> "Toolset":
        return cls((tool for toolset in toolsets for tool in toolset.tools), name=name)

    def executor(self) -> SafeToolExecutor:
        handlers = {
            uri: (lambda arguments, _observations, tool=tool: tool.invoke(arguments))
            for uri, tool in self._by_uri.items()
        }
        return SafeToolExecutor(self.resources, handlers)

    def inspect(self) -> dict[str, object]:
        return {
            "name": self.name,
            "tools": [
                {
                    "uri": resource.uri,
                    "name": resource.name,
                    "side_effect": resource.side_effect_class.value,
                }
                for resource in self.resources
            ],
        }


class _WorkspaceTools:
    """Workspace-bounded implementation behind :func:`default_toolset`."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, value: str = ".") -> Path:
        candidate = (self.root / value).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as error:
            raise PermissionError(f"Path escapes the configured workspace: {value}") from error
        return candidate

    def list_files(self, path: str = ".", pattern: str = "*") -> dict[str, object]:
        """List files below a workspace path using a glob pattern."""

        root = self._path(path)
        values = sorted(
            str(item.relative_to(self.root)).replace("\\", "/")
            for item in root.glob(pattern)
            if item.is_file()
        )
        return {"path": path, "files": values[:1000], "truncated": len(values) > 1000}

    def read_file(self, path: str, start_line: int = 1, end_line: int = 400) -> dict[str, object]:
        """Read a bounded inclusive line range from a UTF-8 workspace file."""

        if start_line <= 0 or end_line < start_line or end_line - start_line > 2000:
            raise ValueError("Expected 1 <= start_line <= end_line with at most 2001 lines.")
        source = self._path(path)
        lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
        selected = lines[start_line - 1:end_line]
        return {
            "path": path,
            "start_line": start_line,
            "end_line": start_line + len(selected) - 1,
            "text": "\n".join(selected),
            "total_lines": len(lines),
        }

    def search_text(self, query: str, path: str = ".", pattern: str = "*.py") -> dict[str, object]:
        """Search bounded text files without invoking an external shell."""

        if not query:
            raise ValueError("Search query cannot be empty.")
        root = self._path(path)
        matches = []
        for source in sorted(root.rglob(pattern)):
            if not source.is_file() or source.stat().st_size > 2_000_000:
                continue
            for line_number, line in enumerate(
                source.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
            ):
                if query.casefold() in line.casefold():
                    matches.append({
                        "path": str(source.relative_to(self.root)).replace("\\", "/"),
                        "line": line_number,
                        "text": line[:500],
                    })
                    if len(matches) >= 500:
                        return {"query": query, "matches": matches, "truncated": True}
        return {"query": query, "matches": matches, "truncated": False}

    def write_file(self, path: str, content: str, overwrite: bool = False) -> dict[str, object]:
        """Write a UTF-8 file inside the workspace after host authorization."""

        target = self._path(path)
        existed = target.exists()
        if existed and not overwrite:
            raise FileExistsError(f"File exists and overwrite is false: {path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")
        return {
            "path": path,
            "bytes": len(content.encode("utf-8")),
            "created": not existed,
        }

    def replace_text(self, path: str, old: str, new: str, count: int = 1) -> dict[str, object]:
        """Replace an exact text fragment in one authorized workspace file."""

        if not old or count <= 0:
            raise ValueError("old text and a positive count are required.")
        target = self._path(path)
        content = target.read_text(encoding="utf-8")
        occurrences = content.count(old)
        if occurrences == 0:
            raise ValueError("Exact text fragment was not found.")
        target.write_text(content.replace(old, new, count), encoding="utf-8", newline="\n")
        return {"path": path, "available_occurrences": occurrences, "replaced": min(count, occurrences)}

    def run_command(self, command: str, cwd: str = ".", timeout_seconds: int = 120) -> dict[str, object]:
        """Run an authorized shell command inside the configured workspace."""

        if not command or timeout_seconds <= 0 or timeout_seconds > 1800:
            raise ValueError("Command and timeout in the range 1..1800 are required.")
        result = subprocess.run(
            command,
            cwd=self._path(cwd),
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            encoding="utf-8",
            errors="replace",
        )
        return {
            "command": command,
            "cwd": cwd,
            "exit_code": result.returncode,
            "stdout": result.stdout[-100_000:],
            "stderr": result.stderr[-100_000:],
        }

    def git_status(self) -> dict[str, object]:
        """Read the concise Git status of the configured workspace."""

        result = subprocess.run(
            ["git", "status", "--short", "--branch"],
            cwd=self.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return {"exit_code": result.returncode, "status": result.stdout, "stderr": result.stderr}


def default_toolset(
    workspace: str | Path,
    *,
    namespace: str = "pra-default",
    tenant_id: str = "default",
) -> Toolset:
    """Return the built-in local coding toolset; execution remains authorized."""

    implementation = _WorkspaceTools(workspace)
    definitions = (
        (implementation.list_files, SideEffectClass.READ, ("files", "workspace")),
        (implementation.read_file, SideEffectClass.READ, ("files", "read")),
        (implementation.search_text, SideEffectClass.READ, ("search", "files")),
        (implementation.git_status, SideEffectClass.READ, ("git", "status")),
        (implementation.write_file, SideEffectClass.WRITE, ("files", "write")),
        (implementation.replace_text, SideEffectClass.WRITE, ("files", "edit")),
        (implementation.run_command, SideEffectClass.WRITE, ("shell", "command")),
    )
    return Toolset(
        (
            Tool(
                function,
                side_effect=side_effect,
                namespace=namespace,
                tenant_id=tenant_id,
                tags=frozenset(tags),
                metadata={"default_toolset": True},
            )
            for function, side_effect, tags in definitions
        ),
        name="pra-default",
    )
