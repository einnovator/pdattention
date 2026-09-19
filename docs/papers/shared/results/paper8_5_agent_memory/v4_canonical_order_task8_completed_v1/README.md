# Canonical-order Task 8 completion

This bundle closes the previously infrastructure-missing eighth exact-prefix
FULL control in the frozen canonical-order campaign. The task is
`django__django-12741`; the model, tokenizer, persistent prefix, seed and FULL
policy are unchanged.

The first retry received no response bytes before the 600-second transport
ceiling. Audit then found two independent ceilings: the selection proxy and the
mini-swe-agent OpenAI client. Commit `04c3a0e0` propagates the declared 1,200
seconds through both. The configuration-invalid intermediate `retry02` was
stopped before its first response. Immutable `retry03` then completed normally:

- 15 calls and 908,437 cumulative message-input tokens;
- primary official submission: unresolved with `patch_apply_failed` because
  the submitted text describes edits rather than emitting a unified diff;
- candidate policy: withheld by the predeclared exact-prefix FULL gate;
- completed campaign: 1/8 primary FULL resolutions, with the only eligible
  M2/P1 arm remaining the exact 0%-saving Task-1 abstention control.

The terminal submission uses only an `echo && echo ...` output chain. Commit
`c1d005a0` certifies this narrow output-only form while continuing to reject
mixed commands, redirection and shell expansion. That permits a separately
labelled grade of the SHA-bound 964-byte terminal workspace patch. The patch
applies and passes both FAIL_TO_PASS tests, but fails 3 of 67 PASS_TO_PASS tests
(64 pass), so the auxiliary workspace outcome is also unresolved. This is a
legitimate task negative, not merely a submission-format artifact or grader
crash.

Important hashes:

- completed campaign state:
  `09862dc96a275673f06252d632d03c693e0355a97538701688d97a651f5b3508`;
- primary official result:
  `dbf769f974306a0a8d1fc364b06efd5ff360ba1abc172b52af146e3beec89197`;
- auxiliary workspace patch:
  `3b6f8b243823f81055fa727980c06f3df254ecb5ff21edc8fe768443398e5357`;
- auxiliary instance report:
  `bfd0be62467766bbf55541cda1386b86e658b092ea22903761544761535bc388`
  (the archived JSON adds a final newline to the remote report bytes).

Primary submission reliability and auxiliary workspace capability remain
separate endpoints. The auxiliary receipt never changes the campaign's primary
official denominator.
