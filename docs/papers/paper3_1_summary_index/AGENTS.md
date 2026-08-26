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

Do not reinterpret the geometry run as a headline result; it contains two
HotpotQA validation identities. Do not start LoRA or downstream generation from
this state without a new, independently powered teacher-headroom result.
