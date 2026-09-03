from types import SimpleNamespace

from pra_hf.agent_workspace import AttachmentManager, HistoryManager, export_session
from pra_hf.context_records import ContextRecord, RecordType
from pra_hf.session_service import AgentSessionState
from pra_hf.tui import AgentShell, CommandRegistry, CommandSpec


def test_command_registry_completion_and_docs():
    registry = CommandRegistry()
    registry.register(CommandSpec("model", "Switch model.", lambda _: None, subcommands=("use",)))
    registry.register(CommandSpec("models", "List models.", lambda _: None))
    assert registry.complete("/mo") == ("/model", "/models")
    assert registry.complete("/model u") == ("use",)
    assert "## `/model`" in registry.markdown()


def test_history_is_bounded_searchable_and_persistent(tmp_path):
    path = tmp_path / "history"
    history = HistoryManager(path, limit=2)
    history.add("one"); history.add("two"); history.add("two"); history.add("three")
    assert history.entries == ["two", "three"]
    assert history.search("thr") == ("three",)
    assert HistoryManager(path).entries == ["two", "three"]


def test_attachment_and_large_paste_records_are_session_scoped(tmp_path):
    records = []
    manager = AttachmentManager(records.append, lambda: tuple(records))
    source = tmp_path / "design.md"; source.write_text("# Design", encoding="utf-8")
    attachment = manager.add(source)
    paste = manager.add_paste("alpha\nbeta")
    assert attachment.mime_type == "text/markdown"
    assert paste.line_count == 2
    assert records[0].record_type == RecordType.ATTACHMENT
    assert records[1].record_type == RecordType.USER_PASTE
    manager.detach(attachment.attachment_id)
    assert [row.attachment_id for row in manager.list()] == [paste.attachment_id]


def test_session_export_separates_transcript_from_binary_metadata(tmp_path):
    state = AgentSessionState("u", "s", records=(
        ContextRecord("m", RecordType.GENERIC_TEXT, {"role": "user", "text": "hello"}),
        ContextRecord("a", RecordType.ATTACHMENT, {"attachment_id": "1", "uri": "file:///a.pdf",
            "name": "a.pdf", "mime_type": "application/pdf", "size_bytes": 4,
            "sha256": "x", "source": "/a.pdf", "active": True}),
    ))
    target = export_session(state, tmp_path / "session.md")
    text = target.read_text(encoding="utf-8")
    assert "## User" in text and "Resource `a`" in text
