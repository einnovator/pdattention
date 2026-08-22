"""Small resource-aware scheduler for independent research trials."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Callable, Generic, Iterable, TypeVar

from .models import ClusterConfig, ResourceRequirements, WorkerConfig

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class ScheduledResult(Generic[T, R]):
    item: T
    workers: tuple[WorkerConfig, ...]
    result: R | None = None
    error: BaseException | None = None


def eligible_workers(
    cluster: ClusterConfig,
    workers: dict[str, WorkerConfig],
    resources: ResourceRequirements,
) -> tuple[WorkerConfig, ...]:
    return tuple(
        worker
        for worker in cluster.resolved_workers(workers)
        if worker.satisfies(resources)
    )


def schedule(
    items: Iterable[T],
    *,
    cluster: ClusterConfig,
    workers: dict[str, WorkerConfig],
    resources_for: Callable[[T], ResourceRequirements],
    execute: Callable[[T, tuple[WorkerConfig, ...]], R],
    fail_fast: bool = False,
) -> list[ScheduledResult[T, R]]:
    """Assign work as capacity becomes available, preserving result input order."""

    pending = list(enumerate(items))
    if not pending:
        return []
    members = cluster.resolved_workers(workers)
    if not members:
        raise ValueError(f"Cluster {cluster.name!r} has no workers.")
    capacity = {worker.name: worker.max_jobs for worker in members}
    parallelism = cluster.max_parallel_trials or sum(capacity.values())
    running: dict[Future, tuple[int, T, tuple[WorkerConfig, ...]]] = {}
    completed: dict[int, ScheduledResult[T, R]] = {}

    def claim(item: T) -> tuple[WorkerConfig, ...] | None:
        resources = resources_for(item)
        needed = resources.workers
        candidates = sorted(
            (
                w
                for w in members
                if w.satisfies(resources)
                and capacity[w.name] > 0
                and (not resources.exclusive or capacity[w.name] == w.max_jobs)
            ),
            key=lambda w: (-w.priority, -capacity[w.name], w.name),
        )
        if len(candidates) < needed:
            return None
        selected = tuple(candidates[:needed])
        for worker in selected:
            capacity[worker.name] -= worker.max_jobs if resources.exclusive else 1
        return selected

    with ThreadPoolExecutor(max_workers=max(1, parallelism)) as pool:
        while pending or running:
            made_progress = True
            while pending and len(running) < parallelism and made_progress:
                made_progress = False
                for position, (index, item) in enumerate(pending):
                    selected = claim(item)
                    if selected is None:
                        continue
                    pending.pop(position)
                    future = pool.submit(execute, item, selected)
                    running[future] = (index, item, selected)
                    made_progress = True
                    break
            if not running:
                item = pending[0][1]
                raise RuntimeError(f"No cluster worker satisfies resources {resources_for(item)!r}.")
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                index, item, selected = running.pop(future)
                for worker in selected:
                    capacity[worker.name] += (
                        worker.max_jobs if resources_for(item).exclusive else 1
                    )
                try:
                    completed[index] = ScheduledResult(item, selected, result=future.result())
                except BaseException as exc:
                    completed[index] = ScheduledResult(item, selected, error=exc)
                    if fail_fast:
                        pending.clear()
    return [completed[index] for index in sorted(completed)]
