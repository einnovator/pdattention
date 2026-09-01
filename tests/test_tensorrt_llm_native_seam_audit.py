from experiments.paper6_4_tensorrt_llm.run_native_seam_audit import classify_sources


def test_stock_paged_cache_and_connector_do_not_open_native_seam() -> None:
    result = classify_sources(
        ("cache_salt_id", "kv_cache_retention_config"),
        {
            "attention_interface": "class AttentionMetadata: kv_cache_params = None",
            "attention_backend": (
                "copy_batch_block_offsets(request_ids)\n"
                "self.kv_cache_block_offsets\n"
                "thop.attention(q)"
            ),
            "kv_connector": "start_load_kv()\nwait_for_layer_load()",
        },
    )

    assert result["official_kv_connector_present"] is True
    assert result["block_table_is_request_owned"] is True
    assert result["one_fused_attention_call"] is True
    assert result["maintainable_narrow_seam"] is False
    assert result["decision"] == "STOP_NO_MAINTAINABLE_NARROW_SEAM"


def test_explicit_request_metadata_and_kernel_contract_open_seam_gate() -> None:
    result = classify_sources(
        ("selected_kv_block_ids",),
        {
            "attention_interface": "selected_kv_block_ids: list[int]",
            "attention_backend": "pra_memory_block_offsets\nthop.attention(q)",
            "kv_connector": "start_load_kv()\nwait_for_layer_load()",
        },
    )

    assert result["maintainable_narrow_seam"] is True
    assert result["decision"] == "NARROW_NATIVE_SEAM_AVAILABLE"
