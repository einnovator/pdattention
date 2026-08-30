from experiments.engine_serving.run_platform_gate_audit import audit


def test_platform_audit_has_explicit_non_inferred_gates() -> None:
    payload = audit()
    assert payload["schema_version"] == "1.0"
    assert isinstance(payload["gates"]["sglang_off_node_backends"], list)
    assert payload["interpretation"]["off_node"] in {
        "READY_FOR_BACKEND_EXPERIMENT",
        "BLOCKED_NO_SUPPORTED_OFF_NODE_BACKEND_INSTALLED",
    }
    assert payload["interpretation"]["cuda_vllm"] in {
        "READY_FOR_CUDA_REPRODUCTION",
        "BLOCKED_REQUIRES_CUDA_COMPUTE_CAPABILITY_8_OR_NEWER",
    }
