from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.paper4_5_agent.runtime_identity import (
    observed_source_checkpoint_revision,
    require_source_checkpoint_revision,
    validate_health_checkpoint_identity,
)


REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"


def test_observes_loaded_config_revision_before_path() -> None:
    config = SimpleNamespace(_commit_hash=REVISION)
    assert observed_source_checkpoint_revision("Qwen/model", runtime_config=config) == REVISION


def test_observes_resolved_snapshot_revision(tmp_path: Path) -> None:
    snapshot = tmp_path / "models--Qwen--model" / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    assert observed_source_checkpoint_revision(snapshot) == REVISION


def test_rejects_configured_alias_without_observed_identity() -> None:
    with pytest.raises(ValueError, match="independently observe"):
        require_source_checkpoint_revision("Qwen/model", REVISION)


def test_rejects_observed_checkpoint_drift(tmp_path: Path) -> None:
    other = "a" * 40
    snapshot = tmp_path / "models--Qwen--model" / "snapshots" / other
    snapshot.mkdir(parents=True)
    with pytest.raises(ValueError, match="does not match"):
        require_source_checkpoint_revision(snapshot, REVISION)


def test_validates_endpoint_observed_revision() -> None:
    assert validate_health_checkpoint_identity(
        {"observed_source_checkpoint_revision": REVISION}, REVISION
    ) == REVISION


def test_rejects_endpoint_without_observed_revision() -> None:
    with pytest.raises(ValueError, match="observed_source_checkpoint_revision"):
        validate_health_checkpoint_identity({"packages": {}}, REVISION)
