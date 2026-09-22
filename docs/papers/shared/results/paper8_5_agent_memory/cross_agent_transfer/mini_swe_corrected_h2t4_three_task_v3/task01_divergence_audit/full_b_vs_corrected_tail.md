# Repeat-control divergence audit

- Instance: `django__django-15277`
- Common visible-message prefix: 16
- First divergent action: 8
- Interpretation: Visible histories were identical before the first divergent assistant action; the fork is backend/model trajectory variability, not selection.

| Run | Actions | Exit | Valid Git diff |
|---|---:|---|---|
| A | 27 | Submitted | True |
| B | 26 | Submitted | True |

## First changed action

- A: `cd /testbed && python -c "
from django.db.models import Value
x = Value('test')
y = x._resolve_output_field()
print('Output field type:', type(y))
print('Validators:', y.validators)
try:
    result = y.clean('1', model_instance=None)
    print('Clean result:', result)
except Exception as e:
    print('Error during clean:', e)
"`
- B: `grep -A 30 "class CharField" /testbed/django/db/models/fields/__init__.py`
