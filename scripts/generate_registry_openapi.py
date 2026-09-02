"""Generate the checked-in PRA Registry OpenAPI contract."""

from __future__ import annotations

import json
from pathlib import Path

from pra_registry.api import create_registry_app
from pra_registry.config import RegistryConfig
from pra_registry.database import RegistryDatabase


def main() -> None:
    app = create_registry_app(
        RegistryConfig(database_url="sqlite:///:memory:"),
        RegistryDatabase("sqlite:///:memory:", create_schema=True),
    )
    destination = Path("docs/site/api/openapi/pra-registry-v1.json")
    destination.write_text(json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
