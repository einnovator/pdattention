from experiments.paper6_4_tensorrt_llm.run_environment_audit import classify_api


def test_connector_does_not_imply_nonprefix_native_attachment() -> None:
    result = classify_api(
        {"cache_salt_id", "kv_cache_retention_config"},
        {"get_cache_block_ids", "pin_blocks", "unpin_blocks_by_id"},
        {
            "register_kv_caches",
            "start_load_kv",
            "wait_for_layer_load",
            "save_kv_layer",
        },
        {
            "build_connector_meta",
            "get_num_new_matched_tokens",
            "update_state_after_alloc",
        },
    )

    assert result["official_connector_interfaces"] is True
    assert result["paged_cache_manager"] is True
    assert result["native_nonprefix_attachment_hook"] is False
    assert result["e2_status"] == "BLOCKED_NO_PUBLIC_NONPREFIX_ATTACHMENT_HOOK"


def test_explicit_nonprefix_hook_opens_the_e2_api_gate() -> None:
    result = classify_api(
        {"selected_kv_block_ids"},
        set(),
        set(),
        set(),
    )
    assert result["native_nonprefix_attachment_hook"] is True
    assert result["e2_status"] == "PUBLIC_NONPREFIX_ATTACHMENT_AVAILABLE"
