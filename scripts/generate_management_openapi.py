"""Generate the checked-in PRA management protocol schema."""

from __future__ import annotations

import json
from pathlib import Path

from pra_hf.management import ManagementAPIConfig, ManagementProvider, create_management_app


def main() -> None:
    app = create_management_app(
        ManagementProvider(engine="engine-adapter", capabilities={"text_fallback": True}),
        ManagementAPIConfig(enabled=True),
    )
    destination = Path("docs/site/api/openapi/pra-management-v1.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(destination)


if __name__ == "__main__":
    main()
