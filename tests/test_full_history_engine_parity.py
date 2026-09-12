from experiments.paper4_5_agent.run_full_history_engine_parity import _turn_gates


def _row(turn: int) -> dict:
    source = 32 + turn * 8
    new = source if turn == 0 else 8
    trace = {
        "full_retention": True,
        "selected_kv_tokens": source,
        "source_tokens": source,
        "source_position_base": source,
        "selected_history_reencoded_tokens": 0,
        "physical_kv_copy": False,
        "physical_kv_copy_bytes": 0,
        "selected_history_kv_copy_bytes": 0,
        "new_history_encoded_tokens": new,
        "canonical_suffix_graft_d2d_bytes": 1024,
        "total_kv_copy_bytes": 2048,
    }
    return {
        "exact": True,
        "trace": trace,
        "response_pra": {"native_attach_bytes": 0},
    }


def test_full_history_gates_separate_selection_from_canonical_copy() -> None:
    gates = _turn_gates([_row(0), _row(1), _row(2)])
    assert all(gates.values())


def test_full_history_gate_rejects_missing_physical_copy_counter() -> None:
    rows = [_row(0), _row(1)]
    del rows[1]["trace"]["physical_kv_copy_bytes"]
    gates = _turn_gates(rows)
    assert gates["byte_exact_outputs"] is True
    assert gates["zero_selected_history_kv_copy"] is False
