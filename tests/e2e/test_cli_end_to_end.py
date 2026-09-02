from __future__ import annotations

import os

import pytest

from experiments.paper4_5_runtime.run_cli_e2e import run


@pytest.mark.e2e
@pytest.mark.skipif(
    os.environ.get("PRA_RUN_CLI_E2E") != "1",
    reason="Set PRA_RUN_CLI_E2E=1 to execute subprocess-level CLI workflows.",
)
def test_every_public_cli_leaf_has_executable_semantic_coverage(tmp_path) -> None:
    report = run(
        tmp_path / "cli_e2e.json",
        live_hub=os.environ.get("PRA_CLI_E2E_LIVE_HUB") == "1",
    )

    assert report["status"] == "PASS"
    assert report["public_leaf_commands"] == report["help_contracts_passed"]
    assert report["public_leaf_commands"] == report["semantic_commands_covered"]
    assert all(receipt["status"] != "FAIL" for receipt in report["receipts"])
