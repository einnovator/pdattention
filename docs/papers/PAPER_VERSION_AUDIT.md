# PRA paper location and version audit

Last verified: **2026-09-14 14:05 Europe/Lisbon**  
Verified publication commit: **`2fe9a230bc52b556056ff24481300dd1024c51a3`**  
Remote: `https://github.com/einnovator/pdattention.git`

## Purpose

The repository has accumulated a publication branch, paper-specific research
branches, integration branches, detached worktrees, and an older aggregate
worktree. A file's modification time, PDF creation date, or convenient local
path is therefore not enough to identify the current manuscript.

This audit records the source-of-truth decision used for each paper. It is a
snapshot, not an automatic guarantee. Update it whenever a paper's canonical
branch changes or a paper is republished to `main`.

## Versioning policy

1. **`main` is the GitHub publication mirror.** Readers should find the current
   selected TeX, PDF, and required generated figures/tables there.
2. **A paper-specific research branch owns ongoing work.** The table below names
   that branch and its intended worktree.
3. **Publication requires an explicit source decision.** Do not choose a version
   solely because its commit, file, or PDF timestamp is newer.
4. **Compare TeX Git blobs first.** Rebuilt PDFs can differ byte-for-byte because
   of timestamps and PDF metadata even when their source and rendered pages are
   equivalent.
5. **Dirty and experimental worktrees are not publication sources** until their
   changes are reviewed, committed, and assigned to a canonical branch.
6. **Raw evidence is not a manuscript revision.** Untracked result JSON or render
   directories must not silently redefine which paper is current.

## Publication map

At this snapshot, every TeX source in this table has the same Git blob in the
listed canonical remote branch and in `origin/main`. PDF paths identify the
published build; PDF binary hashes are deliberately not used for equality.

| Paper | Canonical branch | Audited head | Primary worktree | TeX | PDF | Main source |
|---|---|---:|---|---|---|---|
| 0 | `main` | `2fe9a230` | `D:/git/rd/pdattention-main` | `paper0_position/paper.tex` | `paper0_position/paper.pdf` | MATCH |
| 1 | `main` | `2fe9a230` | `D:/git/rd/pdattention-main` | `paper1_standalone_pra/paper.tex` | `paper1_standalone_pra/paper.pdf` | MATCH; see conflict below |
| 1.5 | `research/paper1-5-rope` | `7acfd884` | `D:/git/rd/pdattention-paper1-5-rope` | `paper1_5_rope/paper.tex` | `paper1_5_rope/paper.pdf` | MATCH |
| 2 | `research/paper2-hf` | `98c4c5f1` | `D:/git/rd/pdattention-paper2` | `paper2_hf/paper.tex` | `paper2_hf/paper.pdf` | MATCH |
| 2.5 | `research/paper2-5-iter-gist` | `cf83aabc` | `D:/git/rd/pdattention-iter-gist` | `paper2_5_iterative_pra/paper.tex` | `paper2_5_iterative_pra/paper.pdf` | MATCH |
| 2.6 | `hybrid-pra` | `06b9f102` | `D:/git/rd/pdattention-hybrid` | `paper2_6_hybrid_pra/paper2_6_v4.tex` | `paper2_6_hybrid_pra/paper2_6_v4.pdf` | MATCH |
| 2.7 | `research/paper2-7-graph-query` | `e44285c5` | `D:/git/rd/pdattention-paper2-7` | `paper2_7_graph_query/paper_2_7.tex` | `paper2_7_graph_query/paper_2_7.pdf` | MATCH |
| 2.8 | `research/paper2-8-qk-compression` | `81fe76d3` | `D:/git/rd/pdattention-paper2-8` | `paper2_8_qk_compression/paper_2_8.tex` | `paper2_8_qk_compression/paper_2_8.pdf` | MATCH |
| 2.9 | `research/paper2-9-look-ahead-back` | `8f83bb48` | no dedicated worktree | `paper2_9_look_ahead_back/paper_2_9.tex` | `paper2_9_look_ahead_back/paper_2_9.pdf` | MATCH |
| 3.1 | `research/paper3-1-summary-index` | `24992aea` | `D:/git/rd/pdattention-summary-index` | `paper3_1_summary_index/paper_3_1.tex` | `paper3_1_summary_index/paper_3_1.pdf` | MATCH |
| 3.2 | `research/paper3-2-rag` | `87d664b9` | `D:/git/rd/pdattention-paper3-2` | `paper3_2/paper_3_2.tex` | `paper3_2/paper_3_2.pdf` | MATCH |
| 3.2 full | `research/paper3-2-rag` | `87d664b9` | `D:/git/rd/pdattention-paper3-2` | `paper3_2/paper_3_2_full.tex` | `paper3_2/paper_3_2_full.pdf` | MATCH |
| 3.3 | `research/paper3-3-crossdoc-retrieval` | `ea3be777` | `D:/git/rd/pdattention-paper3-3` | `paper3_3/paper_3_3.tex` | `paper3_3/paper_3_3.pdf` | MATCH |
| 3.5 | `adaptive-pra` | `a3ba2978` | `D:/git/rd/pdattention-adaptive-pra` | `paper3_5_adaptive_pra/paper.tex` | `paper3_5_adaptive_pra/paper.pdf` | MATCH |
| 3 K/V | `research/paper3-kv-materialization` | `bd4dee9b` | `D:/git/rd/pdattention-kv-materialization` | `paper3_kv_materialization/paper.tex` | `paper3_kv_materialization/paper.pdf` | MATCH |
| 4 | `train` | `9d1d7dce` | `D:/git/rd/pdattention-train` | `paper4_pra_aware_training/paper.tex` | `paper4_pra_aware_training/paper.pdf` | MATCH |
| 4.5 | `research/paper4-5-engine-parity` | `f824e3cc` | `D:/git/rd/pdattention-paper4-5-engine-parity` | `paper4_5_runtime_productization/paper.tex` | `paper4_5_runtime_productization/paper.pdf` | MATCH |
| 5 | `scaling-laws` | `29f924c8` | `D:/git/rd/pdattention-scaling-laws` | `paper5_pra_scaling_laws/paper.tex` | `paper5_pra_scaling_laws/paper.pdf` | MATCH |
| 6.1 | `codex/paper45-m5-engine-parity-local` | `250ab92a` | `D:/git/rd/pdattention-paper45-m5-engine` | `paper6_1_sglang/paper.tex` | `paper6_1_sglang/paper.pdf` | MATCH |
| 6.2 | `codex/paper45-m5-engine-parity-local` | `250ab92a` | `D:/git/rd/pdattention-paper45-m5-engine` | `paper6_2_mlx/paper.tex` | `paper6_2_mlx/paper.pdf` | MATCH |
| 6.3 | `research/paper6-3-openvino` | `7ff19b53` | `D:/git/rd/pdattention-paper6-3-openvino` | `paper6_3_openvino/paper6_3_openvino.tex` | `paper6_3_openvino/paper6_3_openvino.pdf` | MATCH |
| 6.4 | `research/paper6-4-tensorrt-llm` | `4dd35a2b` | `D:/git/rd/pdattention-paper6-4-tensorrt-llm` | `paper6_4_tensorrt_llm/paper6_4_tensorrt_llm.tex` | `paper6_4_tensorrt_llm/paper6_4_tensorrt_llm.pdf` | MATCH |
| 6.5 | `research/paper6-5-tools` | `d4ac55c0` | `D:/git/rd/pdattention-paper6-5` | `paper6_5_tools/paper6_5_v4.tex` | `paper6_5_tools/paper6_5_v4.pdf` | MATCH |
| 6.6 | `research/paper6-6-airllm` | `6387a959` | `D:/git/rd/pdattention-paper6-6-airllm` | `paper6_6_airllm/paper6_6_airllm.tex` | `paper6_6_airllm/paper6_6_airllm.pdf` | MATCH |
| 6.7 | `research/paper6-7-llamacpp` | `85e04242` | `D:/git/rd/pdattention-paper6-7-llamacpp` | `paper6_7_llamacpp/paper6_7_llamacpp.tex` | `paper6_7_llamacpp/paper6_7_llamacpp.pdf` | MATCH |
| 6.8 | `research/paper6-8-ollama` | `f60f9c99` | `D:/git/rd/pdattention-paper6-8-ollama` | `paper6_8_ollama/paper6_8_ollama.tex` | `paper6_8_ollama/paper6_8_ollama.pdf` | MATCH |
| 6.9 | `research/paper6-9-freetokens` | `c3bf8d71` | `D:/git/rd/pdattention-paper6-9-freetokens` | `paper6_9_freetokens/paper6_9_freetokens.tex` | `paper6_9_freetokens/paper6_9_freetokens.pdf` | MATCH |
| 6 vLLM | `codex/paper45-m5-engine-parity-local` | `250ab92a` | `D:/git/rd/pdattention-paper45-m5-engine` | `paper6_vllm/paper.tex` | `paper6_vllm/paper.pdf` | MATCH |
| 7 | `research/paper7-typed-adaptive-context` | `fc2f4154` | `D:/git/rd/pdattention-paper7` | `paper7_records/paper7_typed_adaptive_context_inception.tex` | `paper7_records/paper7_typed_adaptive_context_inception.pdf` | MATCH |
| 8 | `research/paper8-tasks` | `cc2cd4bc` | `D:/git/rd/pdattention-paper8` | `paper8_tasks/paper8_tasks.tex` | `paper8_tasks/paper8_tasks.pdf` | MATCH |
| 8.5 | `research/paper8-5-agent-memory` | `407eede1` | `D:/git/rd/pdattention-paper8-5` | `paper8_5/paper_8_5_draft.tex` | `paper8_5/paper_8_5_draft.pdf` | MATCH |
| 9 | `research/paper9-subagents` | `ac363f0d` | `D:/git/rd/pdattention-paper9` | `paper9_subagents/paper9.tex` | `paper9_subagents/paper9.pdf` | MATCH |

All paths in the TeX and PDF columns are relative to `docs/papers/`.

## Unresolved Paper 1 fork

Two materially different Paper 1 manuscripts currently use the same relative
filename in different worktrees.

| Property | Audited rewrite | Long technical report |
|---|---|---|
| Location | `D:/git/rd/pdattention-main/docs/papers/paper1_standalone_pra/paper.tex` | `D:/git/rd/pdattention/docs/papers/paper1_standalone_pra/paper.tex` |
| Branch | `main` | `distributed-experiments` |
| Last commit touching that branch's Paper 1 | `c4b9a627` on 2026-09-13 | `80cd9f75` on 2026-08-11 |
| TeX Git blob | `a912dcd0e6bb13affd6c4abd2dc644d578683c3b` | `4377d438b1a333ffba4aeb3df3008554c8d80d69` |
| TeX SHA-256 | `a95e33aeaa34f72bab50e47dbd86f5dba8ea45baeb3d56054c2cc441335bef02` | `8e9f1cc025cbea321328a467679bdb6e4ad3aa160161ddc7af37279541f66caf` |
| PDF length | 22 pages | 61 pages |
| Title | *Bounded Sparse Native-KV for Addressable Logical Memory* | *Model-Bounded Sparse Native-KV for Long Context and URI-Addressed Memory* |
| Introduction begins | “Long-context systems face...” | “Progressive Retrieval Attention...” |
| Character | Condensed, later evidence/claim audit | Earlier comprehensive technical report and reimplementation guide |

Important details:

- The 61-page source is also present as the untracked
  `paper1_standalone_pra/paper_rev0.tex` in the `distributed-experiments`
  worktree, and it is byte-identical to that worktree's `paper.tex`.
- The 61-page PDF was rebuilt on 2026-09-14, but rebuilding an older source does
  not make that source the later manuscript revision.
- The audited rewrite has six Paper 1 commits after the common August 11 line.
- The two manuscripts should not continue sharing the same identity. A human
  editorial decision is required: either make the long report canonical, or
  publish it as `paper_technical_report.tex/.pdf` while retaining the audited
  rewrite as `paper.tex/.pdf`.

**Status: unresolved. Do not overwrite either version until this naming and
canonicalization decision is made.**

## Worktree audit and hazards

### `D:/git/rd/pdattention` is not `main`

- Branch: `distributed-experiments` at `c99dbeb6`.
- Upstream divergence: 0 behind / 0 ahead relative to its own upstream.
- Paper-related status entries: **162**.
- Against the 33 TeX sources in the publication index, this worktree contains
  **23 different sources and is missing 10**; none is byte-identical to the
  current `main` source at the same indexed path.
- It also contains copied/untracked paper directories, review artifacts, and
  deleted historical stubs.

Consequently, this path must not be described as a checkout or mirror of
GitHub `main`. Use `D:/git/rd/pdattention-main` for the publication tree.

### Other non-clean or ambiguous worktrees

| Worktree | Branch/state | Paper-related state | Required treatment |
|---|---|---|---|
| `pdattention-paper2-8` | canonical branch, clean tracked files | 1 untracked Paper 2.9 result directory | Review as evidence; not a Paper 2.8 manuscript revision |
| `pdattention-paper3-2` | canonical branch, clean tracked files | 2 untracked result directories | Validate before committing to the Paper 3.2 branch |
| `pdattention-paper4` | `research/paper4-5-runtime`, 20 commits behind upstream | 21 changes including an older edited `paper.tex/.pdf` and render directories | Do not publish; canonical Paper 4.5 is the engine-parity branch |
| `pdattention-paper4-merge` | integration branch, 3 commits behind its configured upstream | no paper changes | Treat as integration scratch, not canonical |
| `pdattention-paper6-3-openvino` | canonical and pushed | one untracked render directory | Ignore/remove render scratch; manuscript is clean |
| `pdattention-paper6-4-tensorrt-llm` | canonical and pushed | one untracked render directory | Ignore/remove render scratch; manuscript is clean |
| `pdattention-paper6-5` | canonical and pushed | one untracked render directory | Ignore/remove render scratch; manuscript is clean |
| `pdattention-paper6-6-airllm` | canonical branch | modified README plus one untracked result JSON | Review evidence/README before the next paper update |
| `pdattention-paper85-token-tail` | experimental branch, 68 behind and 1 ahead of Paper 8.5 | no paper changes | Preserve for experiment provenance; do not publish as Paper 8.5 |
| `pdattention-paper9` | canonical branch | 6 untracked engine-result JSON files | Validate and integrate explicitly before updating Paper 9 |
| `pdattention-scaling-laws` | canonical branch | one untracked result directory | Validate before updating Paper 5 |

The `tmp/` and `tmp_render/` directories under `pdattention-main` are local
render QA scratch and are intentionally untracked.

## Historical and superseded directories on `main`

These directories remain for history but are not primary papers in the current
publication map:

| Directory | Status | Superseded by |
|---|---|---|
| `paper2_hf_integration` | early integration stub; no canonical PDF | Paper 2 and Papers 4.5/6 |
| `paper3_runtime_vllm` | early runtime stub; no canonical PDF | Paper 4.5 and Paper 6 vLLM |
| `paper4_scaling_theory` | early scaling stub; no canonical PDF | Paper 5 |

`paper_pra_survey` is a separate survey manuscript retained on `main`; it is
not numbered in the experimental series.

## Publication verification checklist

Before calling a paper “latest” or copying it to `main`:

1. Fetch remote refs: `git fetch origin --prune`.
2. Record the candidate branch, full commit, worktree, TeX path, and PDF path.
3. Check tracked and untracked changes with
   `git status --short -- docs/papers` in every candidate worktree.
4. Compare source blobs using
   `git rev-parse <ref>:docs/papers/<paper>/<source.tex>`.
5. Review semantic differences when branches disagree; do not resolve them by
   timestamps, line count, or PDF page count.
6. Copy the explicitly selected TeX plus only its required figures, generated
   tables, macros, and bibliography entries.
7. Build from a clean source tree and reject missing files, undefined citations,
   undefined references, and fatal LaTeX errors.
8. Render and inspect representative pages, including dense tables and the last
   page.
9. Commit and push the owning research branch first, then update and push
   `main` as the publication mirror.
10. Verify local HEAD, `origin/main`, and `git ls-remote` agree; update this audit.

## Reconciliation queue

1. Resolve Paper 1's audited-rewrite versus long-report naming and canonical
   publication decision.
2. Retire or clearly archive the stale `research/paper4-5-runtime` worktree.
3. Review uncommitted Paper 3.2, 5, 6.6, and 9 evidence before manuscript use.
4. Give Paper 2.9 a dedicated worktree if active development resumes.
5. Add an automated manifest that verifies branch heads, source blobs, builds,
   and publication paths so this Markdown audit cannot silently become stale.
