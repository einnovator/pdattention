from docker.errors import NotFound

from experiments.paper8_5_agent_memory.swebench_grader_entrypoint import (
    guard_stale_container_scan,
)


def test_stale_global_container_scan_retries_report_without_client():
    clients = []

    def report(
        predictions, full_dataset, run_id, client, namespace,
        instance_image_tag, env_image_tag,
    ):
        clients.append(client)
        if client is not None:
            raise NotFound("stale container")
        return "report.json"

    guarded = guard_stale_container_scan(report)
    result = guarded(
        {"task": {}}, [{"instance_id": "task"}], "run", object(),
        "swebench", "latest", "latest",
    )

    assert result == "report.json"
    assert len(clients) == 2
    assert clients[1] is None


def test_non_docker_reporting_errors_are_not_hidden():
    def report(*args, **kwargs):
        raise ValueError("broken report")

    guarded = guard_stale_container_scan(report)
    try:
        guarded({}, [], "run", object())
    except ValueError as error:
        assert str(error) == "broken report"
    else:
        raise AssertionError("non-Docker reporting failure was hidden")
