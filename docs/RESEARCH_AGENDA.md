# RESEARCH_AGENDA.md

## Vision

Progressive Retrieval Attention treats context as a graph of explicit references rather than a flat token stream. The runtime progressively expands summaries into detailed memory when attention, similarity, uncertainty, or learned gates indicate that additional context is needed.

## Long-Term Goal

Reduce dependence on brute-force long-context windows, one-shot RAG, and some agentic search loops by moving progressive disclosure into transformer inference and runtime memory management.

## Research Phases

1. Position and conceptual framing.
2. Standalone PyTorch proof-of-concept.
3. Hugging Face compatibility wrappers.
4. Native KV-cache and vLLM integration.
5. Scaling, theory, and comparative evaluation.

## Key Research Questions

- Can PRA reduce effective context length while preserving answer quality?
- Which layers benefit from PRA?
- Is cross-attention to retrieved memory sufficient, or is native KV injection required?
- How should RoPE and position encodings be handled for external memory?
- Can recursive anchors outperform one-shot retrieval?
- Can smaller models with PRA rival larger brute-force long-context models on knowledge-intensive tasks?
