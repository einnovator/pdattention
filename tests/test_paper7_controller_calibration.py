from pathlib import Path

from data.full_pra_context_cases import full_pra_context_cases
from experiments.paper7_records.run_controller_calibration import (
    Response,
    _base_prompt,
    _decide,
    _runtime,
)
from pra_hf.progressive_context import ControllerConfig


class _ScriptedController:
    def __init__(self, values):
        self.values = iter(values)

    def chat(self, config, prompt, *, seed, max_tokens=64):
        value = next(self.values)
        return Response(value, str(value), 0.01, 10, 2, False)


def test_hierarchical_controller_executes_second_stage(tmp_path: Path):
    case = next(case for case in full_pra_context_cases() if case.case_id == "c1-billing")
    runtime = _runtime(case, tmp_path)
    config = ControllerConfig("test-model", "D2", protocol="hierarchical")
    client = _ScriptedController((
        {"sufficiency": "NEED_MORE"},
        {"context_action": "MATERIALIZE_FULL"},
    ))

    action, sufficiency, responses = _decide(client, config, case, runtime, 11)

    assert action == "MATERIALIZE_FULL"
    assert sufficiency == "NEED_MORE"
    assert len(responses) == 2
    runtime.runtime.store.close()


def test_model_only_and_adaptive_share_frozen_controller_fingerprint():
    selected = ControllerConfig("llama3.2:3b", "D2", protocol="flat")
    model_only = selected
    adaptive = ControllerConfig(
        selected.model,
        selected.description_level,
        protocol=selected.protocol,
        thinking=selected.thinking,
    )

    assert model_only.fingerprint == adaptive.fingerprint


def test_controller_prompt_omits_host_bound_record_and_cursor_ids(tmp_path: Path):
    case = next(case for case in full_pra_context_cases() if case.case_id == "c3-billing")
    runtime = _runtime(case, tmp_path)
    config = ControllerConfig("test-model", "D2")
    record_id = next(iter(runtime.runtime.records))
    cursor_id = runtime.registry.capabilities[record_id].cursor_id

    prompt = _base_prompt(config, case, runtime)

    assert record_id not in prompt
    assert cursor_id not in prompt
    runtime.runtime.store.close()
