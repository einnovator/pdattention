from experiments.paper8_5_agent_memory.miniswe_semantics import (
    canonicalize_unordered_search_output,
    canonicalize_volatile_filesystem_metadata,
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


def test_long_ls_timestamps_are_canonicalized() -> None:
    output = (
        "total 8\n"
        "drwxrwxrwx 1 root root 4096 Oct  7 12:54 .git\n"
        "-rw-r--r-- 1 root root  807 Aug 13  2025 .editorconfig\n"
    )
    canonical, changed = canonicalize_volatile_filesystem_metadata(
        "ls -la", output
    )
    assert changed
    assert canonical == (
        "total 8\n"
        "drwxrwxrwx 1 root root 4096 <mtime> .git\n"
        "-rw-r--r-- 1 root root  807 <mtime> .editorconfig\n"
    )


def test_plain_or_composed_ls_is_not_canonicalized() -> None:
    output = "drwxr-xr-x 1 root root 4096 Oct  7 12:54 src\n"
    for command in ("ls", "ls -la | head"):
        canonical, changed = canonicalize_volatile_filesystem_metadata(
            command, output
        )
        assert not changed
        assert canonical == output
