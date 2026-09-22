# Repeat-control divergence audit

- Instance: `django__django-15277`
- Common visible-message prefix: 4
- First divergent action: 2
- Interpretation: Visible histories were identical before the first divergent assistant action; the fork is backend/model trajectory variability, not selection.

| Run | Actions | Exit | Valid Git diff |
|---|---:|---|---|
| A | 23 | Submitted | True |
| B | 27 | Submitted | True |

## First changed action

- A: `grep -n "class CharField" /testbed/django/db/models/fields/__init__.py`
- B: `find /testbed -name "*.py" -type f | xargs grep -l "class Value" | head -5`
