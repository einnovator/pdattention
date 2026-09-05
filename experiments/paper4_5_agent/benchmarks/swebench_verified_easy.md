# SWE-bench Verified Easy cohorts

These frozen cohorts provide the first capability gate for Paper 4.5's coding-agent
evaluation. They are derived from the official SWE-bench Verified `test` split at
revision `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`.

| Field | Value |
| --- | --- |
| Upstream | `princeton-nlp/SWE-bench_Verified` |
| Source | https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified |
| License | MIT |
| Split | `test` |
| Difficulty | Exact dataset value `<15 min fix` |
| Eligible tasks | 194 of 500 |
| Official grader | SWE-bench Docker evaluation harness |

## Frozen selection

The generator filters the pinned dataset before any model execution, orders each
eligible ID by

```text
SHA256("paper4.5-swebench-verified-easy-v1" + NUL + instance_id)
```

and takes the first 50. Easy-20 is the first 20 entries of that already frozen
Easy-50 ordering. This avoids repository-order bias while remaining deterministic
and independent of model outcomes.

Regenerate the cards with:

```bash
python -m experiments.paper4_5_agent.build_easy_cohorts
```

The JSON cards contain instance IDs, repositories, base commits, versions, and
difficulty labels. They deliberately exclude problem text, hidden patches, and
grader tests. Their ordered-ID hashes are validated whenever the runner loads a
card.

## Use

Easy-20 is calibration. An official No-PRA result below 20% blocks treatment;
30%-70% is preferred. When it qualifies, Easy-50 becomes the primary paired
No-PRA, pass-through, matched-truncation, and PRA Selected Context cohort.

## Caveats

- Human repair-time difficulty does not guarantee model difficulty.
- Easy-20 and Easy-50 are nested, so they are not independent estimates.
- Calibration and treatment use the same frozen task identities only after the
  admission decision; no failed task may be replaced.
