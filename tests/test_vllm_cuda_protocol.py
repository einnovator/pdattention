from __future__ import annotations

import pytest

from pra_vllm.cuda_protocol import CudaConnectorCommand


def test_cuda_connector_command_round_trip() -> None:
    command = CudaConnectorCommand("load", "tenant.resource-v3", 64)

    assert CudaConnectorCommand.parse(command.cache_salt()) == command
    assert CudaConnectorCommand.parse("ordinary-cache-salt") is None


@pytest.mark.parametrize(
    "mode,key,tokens",
    [("other", "key", 16), ("load", "bad:key", 16), ("store", "key", 0)],
)
def test_cuda_connector_command_rejects_ambiguous_identity(
    mode: str, key: str, tokens: int
) -> None:
    with pytest.raises(ValueError):
        CudaConnectorCommand(mode, key, tokens)

