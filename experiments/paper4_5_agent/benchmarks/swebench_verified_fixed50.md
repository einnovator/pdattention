# SWE-bench Verified fixed-50 benchmark card

## Identity

This is the exact ordered cohort used by the controlled local-model study at
`serbulent-av/agentic-coding`, revision
`8f894c2284b9f73a515024d7c1f32e4d0fb14a04`. The source list is
`benchmarks/agentic-runs/swe-verified/subset_50_instance_ids.txt`. The checked-out
CRLF file hashes to `b12a45dd65c95798a28855c71770545489089ce47db6f4ad1f14ba016084a436`;
the platform-independent LF-joined ID sequence hashes to
`20acb5f7e30fb3c854091e47c4214afb7304a5d47f353408a71ffaa418318131`.

The machine-readable card preserves the source order and contains 50 unique
SWE-bench Verified test IDs from seven repositories. Do not regenerate or
shuffle it. The first ten ordered IDs form the frozen precision diagnostic;
that diagnostic is not a reproduction of the fixed-50 score.
New executions pin dataset revision
`c104f840cc67f8b6eec6f759ebc8b2693d585d4a` in a dedicated cache. The source
study did not report its dataset revision, so historical binary identity
remains unknown even though the instance IDs are exact.

## Published setup

| Field | Pinned value |
| --- | --- |
| Agent | mini-swe-agent `2.4.0` |
| Scaffold | unmodified `swebench_backticks.yaml`, `litellm_textbased` |
| Tool surface | one text/backticks `bash` tool; no native function calling |
| Benchmark harness | SWE-bench `4.1.0`, official Docker grader |
| Engine | vLLM `0.22.1` |
| Model context | 16,384 tokens |
| Step limit | 40 |
| Decoding | greedy, temperature `0` |
| Prefix caching | source default; not explicitly enabled |
| Hardware | one NVIDIA H100 80 GB HBM3 |
| Qwen target | `Qwen/Qwen3-Coder-30B-A3B-Instruct`, 7/50 |
| Gemma target | `google/gemma-4-31B-it`, 19/50 |
| Large-model KV cache | FP8 |

## Reproduction boundary

The source does not publish immutable model/tokenizer revisions, package-file
hashes, Docker image digests, or every retry identity. Those omissions are
recorded separately from local configuration deviations. A run may reproduce
the complete reported configuration and score while pinning its own immutable
revisions, but it cannot claim binary artifact identity with the historical
run without additional provenance from the source authors.

PRA remains locked until the no-PRA baseline has official grading, exact cohort
identity, an accepted score, and no unresolved configuration differences. No
gold patch, hidden test, `FAIL_TO_PASS`, or grader-only metadata may enter the
agent-visible record stream.
