from __future__ import annotations

from pathlib import Path

from experiments.paper6_6_airllm.audit_airllm_environment import audit_source


def test_airllm_source_audit_distinguishes_mlx_from_hf_native(tmp_path: Path) -> None:
    package = tmp_path / "air_llm" / "airllm"
    package.mkdir(parents=True)
    (tmp_path / "air_llm" / "setup.py").write_text("version='3.3.0'\n")
    (package / "airllm_base.py").write_text(
        "from concurrent.futures import ThreadPoolExecutor\n"
        "self.model.forward()\nself.model.generate()\n"
        "module.register_forward_pre_hook(self._pre_hook)\n"
        "module.register_forward_hook(self._post_hook)\n"
        "DynamicCache\nexpert_module\ncompression=True\n"
    )
    (package / "auto_model.py").write_text("AirLLMLlamaMlx\n")
    (package / "airllm_llama_mlx.py").write_text("class AirLLMLlamaMlx: pass\n")
    (package / "airllm_qwen2.py").write_text("")

    mlx = audit_source(tmp_path, device="mlx", hardware="M5", runtime_status="imported")
    cuda = audit_source(tmp_path, device="cuda", hardware="RTX", runtime_status="source")

    assert mlx["airllm_version"] == "3.3.0"
    assert mlx["pra_integration_level"] == "E0"
    assert not mlx["native_hf_pra_available"]
    assert cuda["pra_integration_level"] == "E2_CANDIDATE"
    assert cuda["native_hf_pra_available"]
    assert cuda["weight_streaming_mode"] == "module_pre_hook_load_post_hook_release"
