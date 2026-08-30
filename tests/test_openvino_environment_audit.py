from __future__ import annotations

from experiments.paper6_3_openvino.run_environment_audit import classify_api


def test_audit_does_not_infer_e2_from_scheduler_controls() -> None:
    scheduler = {
        "cache_size",
        "num_kv_blocks",
        "max_num_seqs",
        "max_num_batched_tokens",
        "enable_prefix_caching",
        "dynamic_split_fuse",
        "use_cache_eviction",
    }

    result = classify_api(scheduler, {"generate", "start_chat", "finish_chat"})

    assert result["continuous_batching_ready"] is True
    assert result["e2_status"] == "BLOCKED_NO_PUBLIC_NONPREFIX_ATTACHMENT_HOOK"


def test_audit_recognizes_an_explicit_nonprefix_hook_only() -> None:
    result = classify_api((), {"generate", "attach_nonprefix_kv"})
    assert result["native_nonprefix_attachment_hooks"] == ["attach_nonprefix_kv"]
    assert result["e2_status"] == "PUBLIC_NONPREFIX_ATTACHMENT_AVAILABLE"
