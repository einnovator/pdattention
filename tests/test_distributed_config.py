from pathlib import Path

import pytest

from common.config import (
    apply_overrides,
    discover_yaml_files,
    load_config_sources,
    resolve_infrastructure,
)
from common.distributed.models import DistributionMode


def test_directory_loading_is_recursive_sorted_and_ignores_hidden(tmp_path):
    (tmp_path / "z.yaml").write_text("value: z\nnested: {z: 1}\n", encoding="utf-8")
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "b.yml").write_text("value: b\nnested: {b: 2}\n", encoding="utf-8")
    (tmp_path / ".private.yml").write_text("value: hidden\n", encoding="utf-8")

    files = discover_yaml_files(tmp_path)
    config, sources = load_config_sources(tmp_path)

    assert [item.relative_to(tmp_path).as_posix() for item in files] == ["a/b.yml", "z.yaml"]
    assert config == {"value": "z", "nested": {"b": 2, "z": 1}}
    assert len(sources) == 2


def test_multiple_sources_and_typed_overrides_are_authoritative(tmp_path):
    first = tmp_path / "first.yml"
    second = tmp_path / "second.yml"
    first.write_text("train: {batch_size: 2, enabled: false}\n", encoding="utf-8")
    second.write_text("train: {batch_size: 4}\n", encoding="utf-8")
    config, _ = load_config_sources(first, second)
    apply_overrides(config, ["train.batch_size=8", "train.enabled=true", "seeds=[1, 2]"])
    assert config["train"] == {"batch_size": 8, "enabled": True}
    assert config["seeds"] == [1, 2]


def test_implicit_local_infrastructure_preserves_zero_config_behavior():
    infrastructure = resolve_infrastructure({})
    assert infrastructure.workers["local"].transport == "local"
    assert infrastructure.cluster().name == "local"
    assert infrastructure.clusters["local"].distribution == DistributionMode.LOCAL
    assert infrastructure.storage["local"].type == "local"


def test_bad_references_and_multiple_defaults_fail_clearly():
    with pytest.raises(ValueError, match="unknown worker"):
        resolve_infrastructure({"clusters": {"bad": {"workers": ["missing"]}}})
    with pytest.raises(ValueError, match="Multiple clusters"):
        resolve_infrastructure(
            {
                "clusters": {
                    "a": {"workers": ["local"], "default": True},
                    "b": {"workers": ["local"], "default": True},
                }
            }
        )


def test_legacy_pra_experiment_profiles_are_not_misparsed():
    infrastructure = resolve_infrastructure(
        {"experiments": {"legacy": {"model_name": "tiny", "seeds": [1, 2]}}}
    )
    assert infrastructure.experiments == {}
