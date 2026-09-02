"""Generate the operation catalog page from canonical runtime metadata."""

from pathlib import Path

from pra_control.operations import operation_documentation, tool_documentation


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "site" / "control-plane" / "operation-catalog.md"


def main() -> None:
    OUTPUT.write_text(
        "# Control Plane Operation Catalog\n\n"
        "This page is generated from `pra_control.operations`. The same catalog "
        "drives manager authorization metadata, REST exposure, MCP discovery, "
        "agent capabilities, and audit fields.\n\n"
        "## Operations\n\n" + operation_documentation() + "\n\n"
        "## MCP tools\n\n" + tool_documentation() + "\n\n"
        "Deny patterns override allow patterns. Disabling discovery never grants "
        "or removes a caller permission; manager authorization is always enforced.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
