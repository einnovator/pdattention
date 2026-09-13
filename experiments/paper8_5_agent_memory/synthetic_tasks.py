"""Small diagnostic trajectories with known agent-memory dependencies.

These are not task-quality benchmarks.  They isolate rule activation and
materialization behavior before expensive SWE-bench runs.  Each trajectory is
valid mini-swe-agent-shaped ordinary text and can also be passed to the frozen
replay runner for model-facing smoke tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SyntheticAgentTask:
    task_id: str
    purpose: str
    messages: tuple[Mapping[str, Any], ...]
    probe_decision: int
    reference_command: str
    required_evidence: tuple[str, ...]
    expected_excluded_groups: Mapping[str, tuple[str, ...]]
    materialization_marker: str | None = None

    def trajectory(self) -> dict[str, Any]:
        return {
            "instance_id": self.task_id,
            "messages": [dict(row) for row in self.messages],
            "info": {
                "exit_status": "Submitted",
                "submission": "synthetic-diagnostic-only",
                "evidence_class": "synthetic_mechanism_diagnostic",
                "purpose": self.purpose,
            },
            "synthetic_expectations": {
                "probe_decision": self.probe_decision,
                "reference_command": self.reference_command,
                "required_evidence": list(self.required_evidence),
                "expected_excluded_groups": {
                    key: list(value)
                    for key, value in self.expected_excluded_groups.items()
                },
                "materialization_marker": self.materialization_marker,
            },
        }


_SYSTEM = (
    "You are a coding agent. Inspect the repository, make the smallest correct "
    "change, verify it, and emit exactly one command in a "
    "```mswea_bash_command``` block per turn."
)


def _extra(
    *,
    versions: Mapping[str, str] | None = None,
    post_versions: Mapping[str, str] | None = None,
    verification_resources: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "returncode": 0,
        "cwd": "/workspace",
        "environment_fingerprint": "paper8.5-synthetic-v1",
        "resource_version_fingerprints": dict(versions or {}),
        "post_resource_version_fingerprints": dict(post_versions or {}),
        "verification_resource_ids": list(verification_resources),
        "dependency_resource_ids": list(verification_resources),
        "output_complete": True,
        "timed_out": False,
        "output_truncated": False,
        "tool_semantics": "synthetic-declared-bash-v1",
    }


def _turn(
    thought: str,
    command: str,
    output: str,
    *,
    extra: Mapping[str, Any] | None = None,
    returncode: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = dict(extra or _extra())
    metadata["returncode"] = returncode
    return (
        {
            "role": "assistant",
            "content": (
                f"THOUGHT: {thought}\n"
                f"```mswea_bash_command\n{command}\n```"
            ),
        },
        {
            "role": "user",
            "content": (
                f"<returncode>{returncode}</returncode>\n"
                f"<output>\n{output}\n</output>"
            ),
            "extra": metadata,
        },
    )


def _discovery_consumed() -> SyntheticAgentTask:
    messages: list[Mapping[str, Any]] = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                "In src/normalize.py, make normalize_name strip surrounding "
                "whitespace and lowercase the result. Run tests/test_normalize.py."
            ),
        },
    ]
    messages.extend(_turn(
        "Locate the candidate Python files.",
        "find src -name '*.py' -type f",
        "src/normalize.py\nsrc/legacy_normalize.py",
    ))
    messages.extend(_turn(
        "Read the implementation selected by the task.",
        "cat src/normalize.py",
        "def normalize_name(value):\n    return value",
        extra=_extra(versions={"src/normalize.py": "sha256:norm-v1"}),
    ))
    messages.extend(_turn(
        "Consume the remaining discovery branch before leaving exploration.",
        "cat src/legacy_normalize.py",
        "# compatibility shim\nfrom .normalize import normalize_name",
        extra=_extra(versions={"src/legacy_normalize.py": "sha256:legacy-v1"}),
    ))
    messages.extend(_turn(
        "The discovery is complete; record the stable workspace location.",
        "pwd",
        "/workspace",
    ))
    messages.extend(_turn(
        "Apply the localized one-line fix.",
        "python -c \"from pathlib import Path; p=Path('src/normalize.py'); p.write_text(p.read_text().replace('return value', 'return value.strip().lower()'))\"",
        "",
        extra=_extra(
            versions={"src/normalize.py": "sha256:norm-v1"},
            post_versions={"src/normalize.py": "sha256:norm-v2"},
        ),
    ))
    messages.extend(_turn(
        "Verify the requested behavior.",
        "pytest -q tests/test_normalize.py",
        "1 passed in 0.04s",
        extra=_extra(
            versions={"src/normalize.py": "sha256:norm-v2"},
            verification_resources=("src/normalize.py",),
        ),
    ))
    reference_command = "git diff -- src/normalize.py"
    messages.extend(_turn(
        "Inspect the final patch before submission.",
        reference_command,
        "@@\n-    return value\n+    return value.strip().lower()",
        extra=_extra(versions={"src/normalize.py": "sha256:norm-v2"}),
    ))
    return SyntheticAgentTask(
        "synthetic_discovery_consumed",
        "H1 must activate only after every discovered branch is read and exploration transitions.",
        tuple(messages),
        7,
        reference_command,
        ("normalize_name", "return value", "1 passed"),
        {
            "h1_search_consumed": ("turn:t0000",),
            "h1_all_branches_consumed_strict": ("turn:t0000",),
        },
    )


def _versioned_state_convergence() -> SyntheticAgentTask:
    messages: list[Mapping[str, Any]] = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                "Change src/config.py so DEFAULT_RETRIES is 4, then verify "
                "tests/test_config.py."
            ),
        },
    ]
    messages.extend(_turn(
        "Read the current configuration.",
        "cat src/config.py",
        "DEFAULT_RETRIES = 3\nTIMEOUT_SECONDS = 10",
        extra=_extra(versions={"src/config.py": "sha256:config-v1"}),
    ))
    messages.extend(_turn(
        "Write the requested new value.",
        "python -c \"from pathlib import Path; p=Path('src/config.py'); p.write_text(p.read_text().replace('DEFAULT_RETRIES = 3', 'DEFAULT_RETRIES = 4'))\"",
        "",
        extra=_extra(
            versions={"src/config.py": "sha256:config-v1"},
            post_versions={"src/config.py": "sha256:config-v2"},
        ),
    ))
    current = "DEFAULT_RETRIES = 4\nTIMEOUT_SECONDS = 10"
    messages.extend(_turn(
        "Read back the post-write resource version.",
        "cat src/config.py",
        current,
        extra=_extra(versions={"src/config.py": "sha256:config-v2"}),
    ))
    messages.extend(_turn(
        "Confirm the same current bytes before testing.",
        "cat src/config.py",
        current,
        extra=_extra(versions={"src/config.py": "sha256:config-v2"}),
    ))
    messages.extend(_turn(
        "Verify the changed resource through its focused test.",
        "pytest -q tests/test_config.py",
        "1 passed in 0.03s",
        extra=_extra(
            versions={"src/config.py": "sha256:config-v2"},
            verification_resources=("src/config.py",),
        ),
    ))
    reference_command = "git diff -- src/config.py"
    messages.extend(_turn(
        "Inspect the final patch.",
        reference_command,
        "@@\n-DEFAULT_RETRIES = 3\n+DEFAULT_RETRIES = 4",
        extra=_extra(versions={"src/config.py": "sha256:config-v2"}),
    ))
    return SyntheticAgentTask(
        "synthetic_versioned_state_convergence",
        "H2/H3 must use explicit versions and witnesses rather than command recency alone.",
        tuple(messages),
        6,
        reference_command,
        ("DEFAULT_RETRIES = 4", "1 passed"),
        {
            "h2a_write_current_read": ("turn:t0001",),
            "h2b_verified_write": ("turn:t0001",),
            "h3_read_superseded": ("turn:t0002",),
        },
    )


def _structured_failure_evidence() -> SyntheticAgentTask:
    noise_before = "\n".join(
        f"tests/test_unrelated_{index:02d}.py ." for index in range(36)
    )
    noise_after = "\n".join(
        f"plugin diagnostic {index:02d}: no actionable finding" for index in range(36)
    )
    marker = "E   AssertionError: calculate_total expected 19.80 but got 19.79"
    failure = "\n".join((
        "============================= test session starts =============================",
        noise_before,
        "_______________________ test_discount_rounding ________________________",
        "tests/test_cart.py:42: in test_discount_rounding",
        "    assert calculate_total(items, discount=0.10) == Decimal('19.80')",
        marker,
        "src/cart.py:87: calculate_total",
        "    return sum(round(item.price * Decimal('0.90'), 2) for item in items)",
        "HINT: sum exact discounted values before applying the final currency rounding",
        noise_after,
        "=========================== short test summary info ===========================",
        "FAILED tests/test_cart.py::test_discount_rounding - AssertionError",
        "1 failed, 36 passed in 0.42s",
    ))
    messages: list[Mapping[str, Any]] = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                "Fix calculate_total in src/cart.py. Discounted line items must "
                "be summed exactly and rounded once at the currency boundary."
            ),
        },
    ]
    messages.extend(_turn(
        "Run the focused test to obtain failure evidence.",
        "pytest -q tests/test_cart.py -vv",
        failure,
        extra=_extra(
            versions={"src/cart.py": "sha256:cart-v1"},
            verification_resources=("src/cart.py",),
        ),
        returncode=1,
    ))
    reference_command = "sed -n '78,94p' src/cart.py"
    messages.extend(_turn(
        "Use the traceback location to inspect the implementation.",
        reference_command,
        "def calculate_total(items, discount):\n    return sum(round(item.price * (1 - discount), 2) for item in items)",
        extra=_extra(versions={"src/cart.py": "sha256:cart-v1"}),
    ))
    return SyntheticAgentTask(
        "synthetic_structured_failure_evidence",
        "A decisive traceback in the middle of a long tool result must survive without arbitrary head/tail dumping.",
        tuple(messages),
        2,
        reference_command,
        (marker, "src/cart.py:87", "sum exact discounted values"),
        {},
        marker,
    )


def synthetic_agent_tasks() -> tuple[SyntheticAgentTask, ...]:
    return (
        _discovery_consumed(),
        _versioned_state_convergence(),
        _structured_failure_evidence(),
    )
