from experiments.paper8_5_agent_memory.miniswe_semantics import (
    canonicalize_unordered_search_output,
)


def test_recursive_grep_lines_are_canonicalized() -> None:
    output = "./z.py:hit\n./a.py:hit\n"
    canonical, changed = canonicalize_unordered_search_output(
        'grep -R "hit" -n .', output
    )
    assert changed
    assert canonical == "./a.py:hit\n./z.py:hit\n"


def test_context_search_is_not_reordered() -> None:
    output = "./z.py:hit\ncontext\n./a.py:hit\n"
    canonical, changed = canonicalize_unordered_search_output(
        'grep -R -A 1 "hit" .', output
    )
    assert not changed
    assert canonical == output


def test_composed_search_is_not_reordered() -> None:
    output = "./z.py\n./a.py\n"
    canonical, changed = canonicalize_unordered_search_output(
        "find . -name '*.py' | head", output
    )
    assert not changed
    assert canonical == output
