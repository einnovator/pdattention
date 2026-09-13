# Quarantined Paper 8.5 diagnostics

These artifacts are retained for audit but excluded from scientific result
tables.

- `task02_structured_noop_reconstruction_bug.json`: the structured selector
  retained every line of the sole oversized observation, but the old
  materializer rebuilt the string with `splitlines()` and `"\n".join(...)`.
  That removed trailing-byte detail while reporting the same token count and
  caused 2/11 command divergences. The materializer now returns the canonical
  bytes whenever all lines are selected, with a regression test.
- `task03_h1_current_rule_abstention_control.json`: the requested H1 receipt
  arm produced no H1 candidates under the current stricter rule, so it is an
  exact FULL no-op rather than receipt evidence.

