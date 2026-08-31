from __future__ import annotations

import pytest

from pra_hf.runtime_providers import (
    HFRuntimeProvider,
    MLXRuntimeProvider,
    OpenVINORuntimeProvider,
    RuntimeConfig,
    RuntimeProviderRegistry,
    SGLangRuntimeProvider,
    VLLMRuntimeProvider,
    parse_engine_arguments,
)


def test_builtin_provider_registry_and_unknown_engine() -> None:
    registry = RuntimeProviderRegistry.default()
    assert registry.names() == ("hf", "mlx", "openai", "openvino", "sglang", "vllm")
    with pytest.raises(KeyError, match="Unknown runtime engine"):
        registry.resolve("missing")


def test_engine_capabilities_do_not_overclaim_vllm_scheduler_support() -> None:
    config = RuntimeConfig(engine="vllm", model="org/model")
    vllm = VLLMRuntimeProvider().capabilities(config)
    sglang = SGLangRuntimeProvider().capabilities(RuntimeConfig(engine="sglang"))
    mlx = MLXRuntimeProvider().capabilities(RuntimeConfig(engine="mlx"))

    assert vllm.integration_level.value == "E0"
    assert not vllm.native_kv
    assert not vllm.pra_scheduler
    assert sglang.native_kv and sglang.integration_level.value == "E2"
    assert mlx.native_kv and mlx.integration_level.value == "E2"
    openvino = OpenVINORuntimeProvider().capabilities(
        RuntimeConfig(engine="openvino")
    )
    assert openvino.integration_level.value == "E0"
    assert not openvino.native_kv


def test_provider_build_command_preserves_upstream_escape_hatch() -> None:
    config = RuntimeConfig(
        engine="vllm", model="org/model", host="0.0.0.0", port=9000,
        engine_options={"tensor_parallel_size": 2, "enable_prefix_caching": True},
    )

    command = VLLMRuntimeProvider().build_command(config)

    assert "vllm.entrypoints.openai.api_server" in command
    assert command[command.index("--tensor-parallel-size") + 1] == "2"
    assert "--enable-prefix-caching" in command


def test_parse_engine_arguments_uses_typed_yaml_scalars() -> None:
    assert parse_engine_arguments(("gpu-memory-utilization=0.85", "enabled=true", "count=2")) == {
        "gpu_memory_utilization": 0.85,
        "enabled": True,
        "count": 2,
    }


def test_hf_provider_requires_model_before_managed_launch() -> None:
    with pytest.raises(ValueError, match="model is required"):
        HFRuntimeProvider().build_command(RuntimeConfig())


def test_runtime_doctor_distinguishes_missing_dependency(monkeypatch) -> None:
    monkeypatch.setattr("pra_hf.runtime_providers.importlib.util.find_spec", lambda _: None)

    report = VLLMRuntimeProvider().doctor(RuntimeConfig(engine="vllm"))

    assert report.status == "NOT_INSTALLED"
