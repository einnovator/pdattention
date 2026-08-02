# PRA Dataset Pack

Supplemental dataset folder for the URI-addressed Progressive Retrieval Attention prototype.

Copy or merge this folder into the existing PRA repo, then ask Codex to integrate these datasets into the training/evaluation pipeline.

## Dataset stages

- `stage0_synthetic_memory`: flat synthetic memory blocks and QA.
- `stage1_hierarchical_synthetic`: document/section/paragraph/fact hierarchy.
- `stage2_code_repos`: small synthetic code repository QA.
- `stage3_wikipedia`: placeholder schema and tiny example for Wikipedia-style article/section QA.
- `stage4_books`: public-domain book-style chapter/paragraph QA examples.
- `stage5_technical_docs`: technical documentation QA examples.
- `stage6_github_repos`: repo/directory/file/class/function hierarchy examples.

## Common file types

Each stage may include:

- `documents.jsonl`: source documents/fragments.
- `references.jsonl`: reference handles and URI metadata.
- `questions.jsonl`: QA examples with expected refs/anchors.
- `README.md`: stage-specific explanation.

## Recommended integration

1. Add a dataset loader abstraction:
   - `load_documents(stage_path)`
   - `load_references(stage_path)`
   - `load_questions(stage_path)`
2. Register every row in `references.jsonl` into `ReferenceTable`.
3. Use question prompts containing `<REF_N>` tokens.
4. Resolve URI/anchors through the existing `ReferenceResolver`.
5. Evaluate:
   - answer accuracy
   - selected reference accuracy
   - selected anchor accuracy
   - expansion depth
   - token savings versus full-context baseline
