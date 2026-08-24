# Paper 6.5: Persistent Agent Context

This experiment studies typed tool and skill discovery before native-K/V
materialization. It follows the mechanism-first order in the Paper 6.5 v4
specification.

## Inherited boundary

The branch inherits:

- versioned PRA cache entries and recursive reference resolution;
- bounded long-prompt rollover through the implicit `#__head` reference;
- Qwen, Llama, and Gemma Hugging Face adapters;
- Paper 2.6 token-native and semantic hybrid scoring;
- routing confidence diagnostics and native-K/V materialization budgets.

The branch does not inherit:

- an OpenAI-compatible HTTP serving endpoint;
- a typed agent-resource registry;
- request/namespace/collection discovery-policy hints;
- persistent postings or BM25 catalog indexes;
- confidence-triggered policy escalation and abstention;
- tool/skill catalog, persistent-session, or session-tree benchmarks.

Paper 6.5 adds the missing resource and policy layer without changing the
native-K/V transport. Discovery produces stable resource identities and an
auditable path; selected identities can later resolve to existing PRA cache
entries.

## Stages

1. `M0`: deterministic catalog generation and policy/index evaluation.
2. `M1`: toy decoder with opaque tool identities.
3. `M2`: Qwen3-0.6B bridge for selection and safe call construction.
4. Hierarchical skills, persistent sessions, mutation, and inheritance.
5. OpenAI-compatible serving and maintained harness integration.

Each stage is gated. A later model or execution stage is not evidence for an
earlier discovery claim, and no side-effecting tool executes solely from a
retrieval confidence score.
