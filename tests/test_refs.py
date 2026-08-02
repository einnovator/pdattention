import pytest

from pra_core.references import ReferenceTable
from pra_torch.refs import normalize_ref_tokens, parse_ref_tokens, parse_refs, split_uri_anchor


def test_reference_table_registers_lightweight_tokens():
    table = ReferenceTable()
    first = table.register("mem://doc1#intro", summary="intro summary", metadata={"kind": "doc"})
    second = table.register("search://x?q=y")

    assert first.id == 1
    assert first.token == "<REF_1>"
    assert first.uri == "mem://doc1#intro"
    assert first.summary == "intro summary"
    assert first.metadata == {"kind": "doc"}
    assert second.id == 2
    assert table.get(1) == first
    assert table.find_by_token("<REF_2>") == second
    assert table.all() == [first, second]


def test_parse_ref_tokens_with_table_handles():
    table = ReferenceTable()
    first = table.register("mem://doc1#intro")
    second = table.register("search://x?q=y")

    refs = parse_ref_tokens("See <REF_1> and <REF_2>.", table)

    assert len(refs) == 2
    assert refs[0].token == "<REF_1>"
    assert refs[0].id == 1
    assert refs[0].handle == first
    assert refs[1].handle == second


def test_parse_refs_legacy_syntax_is_deprecated():
    text = "See !!ref:mem://doc1#intro!! and !!ref:search://x?q=y!!."
    with pytest.deprecated_call():
        refs = parse_refs(text)
    assert len(refs) == 2
    assert refs[0].uri == "mem://doc1#intro"
    assert refs[1].uri == "search://x?q=y"


def test_normalize_ref_tokens_preserves_new_tokens_and_collapses_legacy_refs():
    assert normalize_ref_tokens("A <REF_1> B") == "A <REF_1> B"
    assert normalize_ref_tokens("A !!ref:mem://doc1#intro!! B") == "A <REF> B"


def test_split_uri_anchor():
    assert split_uri_anchor("mem://doc1#a.b") == ("mem://doc1", "a.b")
