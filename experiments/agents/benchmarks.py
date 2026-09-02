"""Official benchmark command builders with frozen-cohort enforcement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from .schema import BenchmarkManifest


class BenchmarkAdapter(ABC):
    """Build an official-harness command for exactly one frozen manifest."""

    @abstractmethod
    def command(
        self, manifest: BenchmarkManifest, *, agent: str, model: str, condition: str,
    ) -> list[str]:
        """Return argv without invoking a shell or an unbounded task selection."""

    def _require_tasks(self, manifest: BenchmarkManifest) -> Sequence[str]:
        if not manifest.task_ids:
            raise ValueError("official benchmark execution requires frozen task IDs")
        return manifest.task_ids


class TerminalBenchAdapter(BenchmarkAdapter):
    """Delegate Terminal-Bench 2.1 execution and grading to Harbor."""

    def command(
        self, manifest: BenchmarkManifest, *, agent: str, model: str, condition: str,
    ) -> list[str]:
        del condition
        argv = [
            "harbor", "run", "-d", manifest.dataset, "--agent", agent,
            "--model", model, "-k", str(manifest.repeats),
        ]
        for task_id in self._require_tasks(manifest):
            qualified = task_id if task_id.startswith("terminal-bench/") else f"terminal-bench/{task_id}"
            argv.extend(("-t", qualified))
        return argv


class SWEBenchAdapter(BenchmarkAdapter):
    """Grade agent-produced patches with the official SWE-bench container CLI."""

    def command(
        self, manifest: BenchmarkManifest, *, agent: str, model: str, condition: str,
    ) -> list[str]:
        del agent, model
        argv = [
            "swebench", "eval", manifest.dataset, "-p", "PREDICTIONS.jsonl",
            "--run-id", f"pra-{manifest.name}-{condition}", "-j", "1",
            "--timeout", str(manifest.timeout_seconds),
        ]
        for task_id in self._require_tasks(manifest):
            argv.extend(("-i", task_id))
        return argv


def benchmark_adapter(manifest: BenchmarkManifest) -> BenchmarkAdapter:
    """Resolve only benchmark families whose official harness is supported."""

    if manifest.benchmark == "terminal-bench":
        return TerminalBenchAdapter()
    if manifest.benchmark == "swe-bench":
        return SWEBenchAdapter()
    raise ValueError(f"no external adapter for benchmark {manifest.benchmark!r}")
