# Atomic-E0 cross-task interference and repeatability audit

This bundle tests whether old completed instruction epochs cause interference
in a boundary-free six-episode mini-swe-agent session.  Every counted run uses
the same five-episode prefix, `qwen3-coder:30b-ctx131k`, temperature zero,
seed zero, the ordinary Bash tool, and official SWE-bench grading.  Atomic E0
keeps the active instruction epoch and retires every terminally closed prior
epoch as a whole.  Atomic E2 keeps the active epoch plus the two latest closed
epochs.

## Results

| Control | Official result | Calls | Selected / counterfactual tokens | Own omission | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Django-14089 FULL, raw observations | patch-apply error | 18 | 488,340 / 488,340 | 0% | the model explicitly reports seeing several old PR descriptions and submits prose |
| Django-14089 Atomic E2, raw | unresolved | 30 | 386,671 / 850,681 | 54.55% | retaining two old epochs does not remove the failure |
| Django-14089 Atomic E0, raw repeat 1 | resolved | 9 | 38,639 / 253,118 | 84.73% | active-only retirement can remove cross-task interference |
| Django-14089 Atomic E0, raw repeat 2 | unresolved | 30 | 210,940 / 925,870 | 77.22% | the recovery is not repeatable |
| Django-14089 FULL, normalized long-`ls` metadata | resolved | 22 | 603,290 / 603,290 | 0% | contemporaneous admissible FULL control |
| Django-14089 Atomic E0, matched normalization | unresolved | 30 | 201,895 / 916,825 | 77.98% | strict paired policy loss; failure-aware saving is zero |

The normalized FULL/E0 pair passes the schema-2 execution-identity validator.
Nominal paired token saving is 66.53%, but it is not useful saving because the
candidate loses the FULL success; failure-aware paired saving is therefore
zero and the call delta is +8.

The two raw E0 repeats have byte-identical request 1.  Their request-2 inputs
differ only in volatile `ls -la` timestamps for the parent and `.git`
directories (`12:54` versus `12:56/12:57`).  The assistant then chooses
slightly different search commands and the trajectories separate into a
9-call solve and a 30-call failure.  Commit `daa79a7e` adds a default-off,
manifested matched control that replaces only this long-`ls` timestamp field;
permissions, owners, sizes, filenames, and symlink targets remain visible.
Normalization removes that particular confound, but the matched E0 arm still
fails, proving that active-only retirement is not yet a safe default.

A separate Django-12741 FULL run with `top_k=1` also fails after 19 calls.  It
removes the `using` argument from a method signature but leaves `using=using`
in the body and skips verification.  Tightening the sampler ceiling therefore
does not explain or repair the workload instability.

## Claim boundary

- The raw E0 solve is a recovery diagnostic, not paired preservation evidence.
- The normalized FULL/E0 result is a strict negative and must not be pooled
  with the successful Atomic-E2 preservation cohort.
- Logical omission is not a K/V-residence, latency, or engine-memory claim.
- These results identify cross-task prompt interference and volatile
  observation metadata as real causes, while showing that unsafe within-task
  editing remains an independent failure mode.

