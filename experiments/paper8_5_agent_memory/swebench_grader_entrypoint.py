"""Run the official SWE-bench grader while tolerating stale Docker listings.

Docker Desktop can briefly return a container from ``/containers/json`` and
then return 404 when docker-py inspects that same object. SWE-bench 4.1.0 scans
all containers only while composing its final aggregate report, after the
per-instance evaluation has completed. This wrapper retries that reporting
step without the optional global container/image inventory; it never changes
patch application, test execution, or the per-instance resolution report.
"""

from __future__ import annotations

import runpy
import sys
from typing import Any, Callable


def guard_stale_container_scan(
    make_run_report: Callable[..., Any],
) -> Callable[..., Any]:
    import docker.errors

    def guarded(
        predictions: dict,
        full_dataset: list,
        run_id: str,
        client: Any = None,
        namespace: str | None = None,
        instance_image_tag: str = "latest",
        env_image_tag: str = "latest",
    ) -> Any:
        try:
            return make_run_report(
                predictions, full_dataset, run_id, client, namespace,
                instance_image_tag, env_image_tag,
            )
        except docker.errors.NotFound as error:
            print(
                "PRA_SWEBENCH_REPORT_RETRY_WITHOUT_GLOBAL_DOCKER_INVENTORY: "
                f"{error}",
                file=sys.stderr,
            )
            return make_run_report(
                predictions, full_dataset, run_id, None, namespace,
                instance_image_tag, env_image_tag,
            )

    return guarded


def main() -> None:
    import swebench.harness.reporting as reporting

    reporting.make_run_report = guard_stale_container_scan(
        reporting.make_run_report
    )
    runpy.run_module("swebench.harness.run_evaluation", run_name="__main__")


if __name__ == "__main__":
    main()
