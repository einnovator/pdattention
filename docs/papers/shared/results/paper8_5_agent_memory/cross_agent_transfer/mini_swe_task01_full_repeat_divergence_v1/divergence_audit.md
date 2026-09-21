# Repeat-control divergence audit

- Instance: `django__django-15277`
- Common visible-message prefix: 14
- First divergent action: 7
- Interpretation: Visible histories were identical before the first divergent assistant action; the fork is backend/model trajectory variability, not selection.

| Run | Actions | Exit | Valid Git diff |
|---|---:|---|---|
| A | 27 | Submitted | True |
| B | 17 | Submitted | False |

## First changed action

- A: `grep -A 20 -B 5 "isinstance.*str" /testbed/django/db/models/expressions.py`
- B: `grep -n -A 20 -B 5 "isinstance.*str" /testbed/django/db/models/expressions.py`
