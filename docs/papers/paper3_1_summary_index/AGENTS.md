# Paper 3.1 continuation notes

Follow `docs/AGENTS_3_1.md`. The experiment implementation is
`experiments/paper3_1_summary_index/`; the reusable routing-index boundary is
`src/pra_hf/summary_index.py`.

Do not update numerical claims by hand. Rebuild tables and plots from tracked
row-level artifacts. Validation chooses generation/scoring policy; test
identities are used once for the frozen headline comparison. A summary record
must retain exact `(URI, chunk_id, token span, source SHA-256)` alignment and
must resolve to unchanged native K/V.

The August 2026 pilot freezes these held-out policies:

- HotpotQA: `teacher_8b_generic_1x32_summary_exact`.
- QASPER: `subb_600m_retrieval_1x32_summary_exact`.
- 2Wiki: `teacher_8b_retrieval_1x32_summary_bm25`.
- MuSiQue: `subb_600m_retrieval_1x32_summary_hybrid_a0.50`.

The fixed-budget multi-index continuation is tracked under
`shared/results/paper3_1_summary_index/multi_index/`. It freezes RRF/fusion
choices on eight validation identities and evaluates 24 held-out identities.
Summaries have unique recoveries but no resolved positive marginal value after
`L+E+QK`; the default stack therefore excludes `S`.

Do not reinterpret the geometry run as a headline result; it contains two
HotpotQA validation identities. Do not start LoRA or downstream generation from
this state without a new, independently powered marginal-recall result beyond
the lexical/extractive/QK stack.
