import json

import pytest
import torch

from common.distributed.launcher import launch_local
from common.distributed.smoke import ddp_rank


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed unavailable")
def test_two_process_cpu_gloo_smoke(tmp_path):
    output = tmp_path / "ddp.json"
    launch_local(ddp_rank, world_size=2, args=(str(output),))
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["world_size"] == 2
    assert len(payload["weight"][0]) == 2
