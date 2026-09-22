# Pi exact-token H2/T4 Task 1 pair

Pi 0.75.3 and `qwen3-coder:30b` ran the locked `django__django-15277`
task with the exact tokenizer and fixed generation parameters. The fresh FULL
control resolved officially in 28 model calls with the minimal 689-byte patch.
The corrected H2/T4 tail-90 arm did not resolve.

The candidate saved 8.53% of materialized ordinary-text input relative to its
own trajectory, but only 0.30% came from whole-record retirement. Most apparent
saving came from rewriting one oversized source-view observation inside a
selected causal group. After the first changed input, Pi diverged from the
FULL action sequence, failed three edits, overwrote the complete source file
with a fragment, and submitted a 91,283-byte destructive patch.

This is a failed policy-transfer result, not an engine result. It motivated the
`declared_safe` boundary-compaction mode: record-internal rewriting is disabled
unless the record contract explicitly says that stable, natural child spans
were created before K/V encoding. Paper 4.5 may map only whole-record omission,
or such pre-segmented child-span omission, to resident K/V savings.

The raw 2.3 MB Pi event streams and proxy traces remain in the immutable run
directory on the Medium Mac. `qualification.json`, `model_identity.json`, and
`divergence_audit.json` are the compact review artifacts.
