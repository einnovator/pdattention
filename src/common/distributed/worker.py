"""Worker capability discovery used by CLI health checks."""

from __future__ import annotations

import json
import platform
import sys

from .models import WorkerConfig
from .transport import transport_for


_PROBE = (
    "import json,platform; "
    "import torch; "
    "print(json.dumps({'python':platform.python_version(),"
    "'platform':platform.platform(),'torch':torch.__version__,"
    "'cuda':torch.cuda.is_available(),'cuda_devices':torch.cuda.device_count(),"
    "'mps':bool(getattr(torch.backends,'mps',None) and torch.backends.mps.is_available())}))"
)


def ping_worker(worker: WorkerConfig) -> dict:
    """Probe Python, PyTorch, and accelerator availability through its transport."""

    executable = worker.python_executable if worker.transport == "ssh" else sys.executable
    result = transport_for(worker).run([executable, "-c", _PROBE], timeout=worker.timeout_seconds or 30)
    if result.returncode:
        return {"name": worker.name, "ok": False, "error": result.stderr.strip()}
    try:
        capabilities = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        return {"name": worker.name, "ok": False, "error": f"Invalid probe output: {exc}"}
    return {"name": worker.name, "ok": True, **capabilities}
