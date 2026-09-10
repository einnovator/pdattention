"""Opt-in Docker SDK workaround for an uninspectable daemon list entry.

Install this file as ``sitecustomize.py`` and set
``PAPER45_SKIP_PHANTOM_DOCKER_CONTAINERS=1``. The workaround changes only
Docker's high-level container enumeration: if the daemon lists an ID that it
then reports as missing, the missing entry is omitted. SWE-bench test execution
and result parsing are not modified.
"""

from __future__ import annotations

import os
import sys


if os.environ.get("PAPER45_SKIP_PHANTOM_DOCKER_CONTAINERS") == "1":
    from docker.errors import NotFound
    from docker.models.containers import ContainerCollection

    _original_list = ContainerCollection.list

    def _list_without_phantoms(self, *args, **kwargs):
        try:
            return _original_list(self, *args, **kwargs)
        except NotFound:
            sparse_kwargs = dict(kwargs)
            sparse_kwargs["sparse"] = True
            candidates = _original_list(self, *args, **sparse_kwargs)
            valid = []
            for container in candidates:
                try:
                    container.reload()
                except NotFound:
                    print(
                        "PAPER45_SKIPPED_PHANTOM_DOCKER_CONTAINER="
                        f"{container.id}",
                        file=sys.stderr,
                    )
                else:
                    valid.append(container)
            return valid

    ContainerCollection.list = _list_without_phantoms
