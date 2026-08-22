import pytest

from common.distributed.models import (
    ClusterConfig,
    DistributionMode,
    ResourceRequirements,
    WorkerConfig,
)
from common.distributed.scheduler import schedule


def test_distribution_aliases_and_resource_matching():
    assert DistributionMode.from_value("multi-seed") == DistributionMode.SEEDS
    worker = WorkerConfig.from_mapping(
        "gpu", {"device": "cuda:0", "memory_gb": 16, "tags": ["fast", "cuda"]}
    )
    assert worker.satisfies(ResourceRequirements(device="cuda", min_memory_gb=12, tags=("fast",)))
    assert not worker.satisfies(ResourceRequirements(min_memory_gb=20))
    assert not worker.satisfies(ResourceRequirements(tags=("missing",)))


def test_cluster_rejects_storage_weight_updates_for_ddp():
    with pytest.raises(ValueError, match="synchronous weight"):
        ClusterConfig.from_mapping(
            "bad", {"workers": ["local"], "distribution": "ddp", "weight_update_transport": "storage"}
        )


def test_scheduler_observes_capacity_and_returns_input_order():
    workers = {
        "a": WorkerConfig.from_mapping("a", {"transport": "process", "max_jobs": 1}),
        "b": WorkerConfig.from_mapping("b", {"transport": "process", "max_jobs": 1}),
    }
    cluster = ClusterConfig.from_mapping("two", {"workers": ["a", "b"], "max_parallel_trials": 2})
    results = schedule(
        [0, 1, 2, 3],
        cluster=cluster,
        workers=workers,
        resources_for=lambda _: ResourceRequirements(),
        execute=lambda value, selected: (value, selected[0].name),
    )
    assert [result.result[0] for result in results] == [0, 1, 2, 3]
    assert {result.result[1] for result in results} == {"a", "b"}


def test_scheduler_reports_unsatisfied_resources():
    workers = {"cpu": WorkerConfig.from_mapping("cpu", {"device": "cpu"})}
    cluster = ClusterConfig.from_mapping("cpu", {"workers": ["cpu"]})
    with pytest.raises(RuntimeError, match="No cluster worker"):
        schedule(
            [1],
            cluster=cluster,
            workers=workers,
            resources_for=lambda _: ResourceRequirements(device="cuda"),
            execute=lambda *_: None,
        )
