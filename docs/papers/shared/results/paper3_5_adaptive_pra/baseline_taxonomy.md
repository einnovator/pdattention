# Baseline taxonomy

The controlled benchmark separates mechanisms that are often conflated:

| Family | Representative rows | What is active | Status in this study |
|---|---|---|---|
| Dense/native long context | full context, truncation, matched budget | prompt/native K/V | controlled matched-corpus proxy |
| External text retrieval | single-shot and multi-query RAG | retrieved text retokenized in prompt | controlled matched-corpus proxy |
| Iterative retrieval | iterative RAG | text reached after a second retrieval step | controlled matched-corpus proxy |
| K/V eviction or compression | H2O, SnapKV, StreamingLLM, ClusterKV | subset/summary of an existing sequence cache | mechanism-faithful selector proxy, not third-party kernels |
| PRA | fixed/adaptive native-Q/K search | contextual native K/V selected from addressable memory | package implementation plus controlled corpus |
| Sparse architectural attention | Longformer, BigBird, Reformer, Routing Transformer | checkpoint-specific sparse pattern | taxonomy only; no compatible checkpoint comparison |
| Recurrent/persistent memory | Compressive Transformer, Infini-Attention, Memorizing Transformer, RETRO, Titans | recurrent state or external datastore | taxonomy only; no compatible checkpoint comparison |

The proxy rows test accounting and frontier construction. They are not substitutes
for upstream H2O/SnapKV/ClusterKV kernels or a production RAG stack. The paper marks
those external integrations as pending and makes no production-speed claim from
the standalone Python page table or gather paths.
