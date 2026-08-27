from __future__ import annotations

import pytest

from pra_hf import ExecutionAuthorization, SideEffectClass, Tool, Toolset, default_toolset


def test_toolset_preserves_declared_side_effect_and_executes_mapping() -> None:
    def lookup(name: str) -> dict[str, str]:
        """Look up one name."""

        return {"name": name}

    toolset = Toolset((Tool(lookup, side_effect=SideEffectClass.READ),))
    resource = toolset.resources[0]
    outcome = toolset.executor().execute(
        type("Call", (), {"name": "lookup", "arguments": {"name": "Ada"}})(),
        selected_uris=(resource.uri,),
        authorization=ExecutionAuthorization(frozenset((resource.uri,))),
        call_id="call-1",
    )

    assert resource.side_effect_class == SideEffectClass.READ
    assert outcome.output == {"name": "Ada"}


def test_default_toolset_is_workspace_bounded_and_write_authorized(tmp_path) -> None:
    (tmp_path / "note.txt").write_text("alpha", encoding="utf-8")
    toolset = default_toolset(tmp_path)
    by_name = {resource.name: resource for resource in toolset.resources}
    executor = toolset.executor()

    read = executor.execute(
        type("Call", (), {"name": "read_file", "arguments": {"path": "note.txt"}})(),
        selected_uris=(by_name["read_file"].uri,),
        authorization=ExecutionAuthorization(frozenset((by_name["read_file"].uri,))),
        call_id="read-1",
    )
    denied = executor.execute(
        type("Call", (), {"name": "write_file", "arguments": {"path": "new.txt", "content": "x"}})(),
        selected_uris=(by_name["write_file"].uri,),
        authorization=ExecutionAuthorization(frozenset((by_name["write_file"].uri,))),
        call_id="write-1",
    )

    assert read.output["text"] == "alpha"
    assert denied.reason == "write_not_authorized"
    with pytest.raises(PermissionError):
        next(tool for tool in toolset.tools if tool.record.name == "read_file").function("../outside.txt")


def test_toolset_rejects_duplicate_names() -> None:
    def same() -> dict[str, object]:
        return {}

    with pytest.raises(ValueError, match="unique"):
        Toolset((Tool(same), Tool(same, namespace="another")))
