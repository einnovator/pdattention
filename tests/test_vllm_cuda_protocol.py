from __future__ import annotations

import pytest

from pra_vllm.cuda_protocol import CudaConnectorCommand


def test_cuda_connector_command_round_trip() -> None:
    command = CudaConnectorCommand("load", "tenant.resource-v3", 64)
    hot = CudaConnectorCommand("load", "tenant.resource-v3", 64, residency="hot")
    scoped = CudaConnectorCommand(
        "load", "tenant.resource-v3", 64, request_scope="request-7"
    )

    assert CudaConnectorCommand.parse(command.cache_salt()) == command
    assert CudaConnectorCommand.parse(hot.cache_salt()) == hot
    assert CudaConnectorCommand.parse(scoped.cache_salt()) == scoped
    assert hot.cache_salt().startswith("pra-cuda-v2:load:hot:")
    assert CudaConnectorCommand.parse("ordinary-cache-salt") is None


@pytest.mark.parametrize(
    "mode,key,tokens,residency,scope",
    [
        ("other", "key", 16, "warm", None),
        ("load", "bad:key", 16, "warm", None),
        ("store", "key", 0, "warm", None),
        ("load", "key", 16, "cold", None),
        ("store", "key", 16, "hot", None),
        ("load", "key", 16, "warm", "bad:scope"),
    ],
)
def test_cuda_connector_command_rejects_ambiguous_identity(
    mode: str, key: str, tokens: int, residency: str, scope: str | None
) -> None:
    with pytest.raises(ValueError):
        CudaConnectorCommand(
            mode, key, tokens, residency=residency, request_scope=scope
        )
