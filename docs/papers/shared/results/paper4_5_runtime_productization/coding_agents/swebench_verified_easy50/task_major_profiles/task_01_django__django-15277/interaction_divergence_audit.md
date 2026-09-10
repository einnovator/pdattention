# Task interaction divergence audit

Baseline: `plain_clean_receipt` (30 actions).

| Arm | Score | Actions | Exact responses | First response diff | Exact commands | First command diff |
|---|---:|---:|---:|---:|---:|---:|
| `pra_100_liveprefix` | 1.0 | 30 | 30 | none | 30 | none |
| `pra_75_active_tail` | 0.0 | 23 | 2 | 3 | 3 | 4 |
| `pra_50_active_tail` | 0.0 | 50 | 2 | 3 | 3 | 4 |
| `pra_25_active_tail` | 0.0 | 50 | 2 | 3 | 3 | 4 |
| `pra_50_cache_off` | 0.0 | 44 | 2 | 3 | 3 | 4 |
| `pra_75_progress_spine_v4` | 1.0 | 33 | 4 | 5 | 4 | 5 |
| `pra_50_progress_spine_v4` | 0.0 | 50 | 4 | 5 | 4 | 5 |

## Main finding

The 100% PRA arm is response-for-response exact with the successful plain run, which qualifies the engine and transport path when no history is removed.

Every legacy v3 reduced-context arm first changes the executed plan at action 4 and exposes the same selected records at that point. The selector keeps the five structural task segments and the active tail, but drops the first two completed action/observation bundles. Nominal 75%, 50%, and 25% profiles therefore collapse to the same early prompt because the structural floor dominates the budget.

At that request the whitespace accounting is logical=1600, mandatory-tail=136, selected-resource=1259, and physical=1395 tokens. The small early saving is purchased by deleting the agent's progress state.

The legacy v3 failure mode is a progress-memory loop, not a template or K/V attachment fault: retrieval is queried only with the latest tool observation and repeatedly surfaces older, lexically similar CharField inspections. It does not preserve a stable plan, the latest successful mutation, verification state, or the patch submission lifecycle.

The v4 progress-spine repair pins the two latest completed causal turns plus the latest mutation and verification turns. In `pra_75_progress_spine_v4`, its first command difference is action 5. The 75% arm takes a narrower but semantically equivalent source view, later recovers from a stale-line edit, creates the patch at action 31, submits exactly at action 33, and resolves the task. Its estimated logical-context saving is 0.22320820500310026.

The v4 progress-spine repair pins the two latest completed causal turns plus the latest mutation and verification turns. In `pra_50_progress_spine_v4`, its first command difference is action 5. The 50% arm retains the exact earlier CharField path but does not use it, launches redundant whole-tree searches, corrupts the method boundary while editing, and reaches the 50-step limit without a submitted patch. This is a consumption/edit-execution failure despite the required state being selected. Its estimated logical-context saving is 0.45903509266039544.

## pra_75_active_tail: first divergence

The first differing executed command is action 4.
Selected resource IDs: `['m1-0-user', 'm1-1-user', 'm1-2-user', 'm1-3-user', 'm1-4-user']`.
Omitted historical candidate IDs: `['m2-0-assistant', 'm3-0-user', 'm4-0-assistant', 'm5-0-user']`.

Plain command: `find ./django/db/models -name "*.py" | xargs grep -l "class Value" | head -5`

Treated command: `grep -n "class CharField" ./django/db/models/fields/__init__.py`

Most repeated later commands: `grep -n "class CharField" ./django/db/models/fields/__init__.py` (3x); `grep -A 5 -B 5 "if self.max_length is not None" ./django/db/models/fields/__init__.py` (2x)

Milestones: Value inspection=None; source mutation=9; git diff=None; patch file=None; exact submission=None.

## pra_50_active_tail: first divergence

The first differing executed command is action 4.
Selected resource IDs: `['m1-0-user', 'm1-1-user', 'm1-2-user', 'm1-3-user', 'm1-4-user']`.
Omitted historical candidate IDs: `['m2-0-assistant', 'm3-0-user', 'm4-0-assistant', 'm5-0-user']`.

Plain command: `find ./django/db/models -name "*.py" | xargs grep -l "class Value" | head -5`

Treated command: `grep -n "class CharField" ./django/db/models/fields/__init__.py`

Most repeated later commands: `grep -n "class CharField" ./django/db/models/fields/__init__.py` (6x); `sed -n '1007,1025p' ./django/db/models/fields/__init__.py` (4x); `sed -n '1007,1020p' ./django/db/models/fields/__init__.py` (4x)

Milestones: Value inspection=None; source mutation=18; git diff=None; patch file=None; exact submission=None.

## pra_25_active_tail: first divergence

The first differing executed command is action 4.
Selected resource IDs: `['m1-0-user', 'm1-1-user', 'm1-2-user', 'm1-3-user', 'm1-4-user']`.
Omitted historical candidate IDs: `['m2-0-assistant', 'm3-0-user', 'm4-0-assistant', 'm5-0-user']`.

Plain command: `find ./django/db/models -name "*.py" | xargs grep -l "class Value" | head -5`

Treated command: `grep -n "class CharField" ./django/db/models/fields/__init__.py`

Most repeated later commands: `grep -n "class CharField" ./django/db/models/fields/__init__.py` (9x); `grep -n "class Value" ./django/db/models/expressions.py` (4x); `grep -n -A 20 "class BinaryField" ./django/db/models/fields/__init__.py` (3x)

Milestones: Value inspection=19; source mutation=44; git diff=None; patch file=None; exact submission=None.

## pra_50_cache_off: first divergence

The first differing executed command is action 4.
Selected resource IDs: `['m1-0-user', 'm1-1-user', 'm1-2-user', 'm1-3-user', 'm1-4-user']`.
Omitted historical candidate IDs: `['m2-0-assistant', 'm3-0-user', 'm4-0-assistant', 'm5-0-user']`.

Plain command: `find ./django/db/models -name "*.py" | xargs grep -l "class Value" | head -5`

Treated command: `cat ./django/db/models/fields/__init__.py | grep -A 20 -B 5 "class CharField"`

Most repeated later commands: `cat ./django/db/models/fields/__init__.py | grep -A 20 "class BinaryField"` (3x); `grep -n "class CharField" ./django/db/models/fields/__init__.py` (3x); `cat ./django/db/models/fields/__init__.py | grep -A 50 "class CharField"` (2x)

Milestones: Value inspection=None; source mutation=24; git diff=42; patch file=42; exact submission=44.

## pra_75_progress_spine_v4: first divergence

The first differing executed command is action 5.
Selected resource IDs: `['m1-0-user', 'm1-1-user', 'm1-2-user', 'm1-3-user', 'm1-4-user', 'm4-0-assistant', 'm5-0-user', 'm6-0-assistant', 'm7-0-user']`.
Omitted historical candidate IDs: `['m2-0-assistant', 'm3-0-user']`.

Plain command: `cat ./django/db/models/fields/__init__.py | grep -A 50 "class CharField" | head -60`

Treated command: `cat ./django/db/models/fields/__init__.py | grep -A 20 "class CharField"`

Most repeated later commands: `python -c "
from django.db.models import Value
x = Value('test')
y = x._resolve_output_field()
print('Validators:', y.validators)
try:
    y.clean('1', model_instance=None)
    print('Clean succeeded')
except Exception as e:
    print('Clean failed with error:', type(e).__name__, str(e))
"` (2x)

Milestones: Value inspection=4; source mutation=19; git diff=31; patch file=31; exact submission=33.

## pra_50_progress_spine_v4: first divergence

The first differing executed command is action 5.
Selected resource IDs: `['m1-0-user', 'm1-1-user', 'm1-2-user', 'm1-3-user', 'm1-4-user', 'm4-0-assistant', 'm5-0-user', 'm6-0-assistant', 'm7-0-user']`.
Omitted historical candidate IDs: `['m2-0-assistant', 'm3-0-user']`.

Plain command: `cat ./django/db/models/fields/__init__.py | grep -A 50 "class CharField" | head -60`

Treated command: `cat ./django/db/models/fields/__init__.py | grep -A 20 "class CharField"`

Most repeated later commands: `python -c "
from django.db.models import Value
x = Value('test')
y = x._resolve_output_field()
print('Validators:', y.validators)
try:
    y.clean('1', model_instance=None)
    print('Clean succeeded')
except Exception as e:
    print('Error:', type(e).__name__, str(e))
"` (3x); `sed -n '1010,1020p' /testbed/django/db/models/fields/__init__.py` (2x); `sed -n '1007,1030p' /testbed/django/db/models/fields/__init__.py` (2x)

Milestones: Value inspection=4; source mutation=22; git diff=None; patch file=None; exact submission=None.
