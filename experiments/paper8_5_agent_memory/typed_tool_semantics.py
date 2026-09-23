"""Portable typed-tool semantics shared by policy and evidence reducers."""

DEFAULT_TOOL_SEMANTICS = {
    "read": {
        "category": "filesystem",
        "operation_kind": "read",
        "resource_arguments": ["path", "file_path", "filePath"],
    },
    "read_file": {
        "category": "filesystem",
        "operation_kind": "read",
        "resource_arguments": ["path", "file_path", "filePath"],
    },
    "grep": {
        "category": "filesystem",
        "operation_kind": "read",
        "resource_arguments": ["path"],
    },
    "search_text": {
        "category": "filesystem",
        "operation_kind": "read",
        "resource_arguments": ["path"],
    },
    "find": {"category": "filesystem", "operation_kind": "search_discovery"},
    "glob": {"category": "filesystem", "operation_kind": "search_discovery"},
    "ls": {"category": "filesystem", "operation_kind": "search_discovery"},
    "edit": {
        "category": "filesystem",
        "operation_kind": "write",
        "resource_arguments": ["path", "file_path", "filePath"],
    },
    "write": {
        "category": "filesystem",
        "operation_kind": "write",
        "resource_arguments": ["path", "file_path", "filePath"],
    },
    "apply_patch": {
        "category": "filesystem",
        "operation_kind": "write",
        "resource_arguments": ["path", "file_path", "filePath"],
    },
    "replace_text": {
        "category": "filesystem",
        "operation_kind": "write",
        "resource_arguments": ["path", "file_path", "filePath"],
    },
    # Arbitrary shell remains unknown unless execution middleware supplies a
    # complete receipt. This is a safety barrier, not agent-specific logic.
    "bash": {"category": "shell", "operation_kind": "unknown"},
    "run_command": {"category": "shell", "operation_kind": "unknown"},
}
