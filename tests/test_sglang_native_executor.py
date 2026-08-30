from dataclasses import dataclass

import pytest

from pra_hf.deployment import PRAWireBudget, PRAWireRequest, PRAWireResource
from pra_hf.engine_memory import LogicalPRABlockStore
from pra_sglang.native_executor import SGLangInProcessNativeExecutor


@dataclass
class _Runner:
    model: object = object()


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(range(len(text.split())))


def _executor() -> SGLangInProcessNativeExecutor:
    value = object.__new__(SGLangInProcessNativeExecutor)
    value.runner = _Runner()
    value.tokenizer = _Tokenizer()
    value.model_id = "model"
    value.model_revision = "revision"
    value.block_store = LogicalPRABlockStore()
    return value


def test_selected_resources_and_shareable_identity_are_stable() -> None:
    executor = _executor()
    resource = PRAWireResource(
        "resource",
        "pra://tenant/resource",
        text="one two three",
        metadata={"version": "v1", "shareable": True},
    )
    request = PRAWireRequest(
        model="model",
        messages=({"role": "user", "content": "question"},),
        tenant_id="tenant",
        session_id="session-a",
        resources=(resource,),
        budget=PRAWireBudget(max_resources=1, max_selected_tokens=8),
        pra_policy={"selected_resource_ids": ["resource"]},
    )
    other_session = PRAWireRequest(
        **{**request.__dict__, "session_id": "session-b"}
    )

    assert executor._selected_resources(request) == (resource,)
    assert executor._resource_tokens(resource) == [0, 1, 2]
    assert executor._logical_key(request, resource) == executor._logical_key(
        other_session, resource
    )


def test_token_ids_normalizes_mapping_array_and_single_batch() -> None:
    class Array:
        def tolist(self):
            return [[1, 2, 3]]

    assert SGLangInProcessNativeExecutor._token_ids({"input_ids": Array()}) == [1, 2, 3]


def test_token_ids_rejects_multiple_prompt_batches() -> None:
    with pytest.raises(ValueError, match="more than one"):
        SGLangInProcessNativeExecutor._token_ids([[1], [2]])
