# AGENTS.md — Paper 6.8: PRA with Ollama

## PRA as a transparent local-model product integration

### Central question
Can existing Ollama applications transparently move from E0 gateway use to deeper PRA integration through backend capability negotiation, without turning Ollama into a second PRA runtime?

This is primarily a product/runtime integration paper, not another attention-mechanism paper.

### Architectural rule
Reuse native PRA work from the actual Ollama execution backend where possible, especially llama.cpp-derived/native paths. Do not maintain a second PRA attention implementation merely for Ollama.

PRA owns typed records, selection, gateway, profiles, sessions/tasks, storage policy and qualification. Ollama owns local model lifecycle, packaging, API/server UX, backend selection and scheduling.

### Work plan
1. Pin Ollama version and audit current API/OpenAI compatibility, runner/backend architecture, llama.cpp relationship, model lifecycle, keep-alive, context/cache ownership, scheduling/concurrency and extension seams. Do not assume historical architecture.
2. Establish E0 gateway integration for ordinary Ollama/OpenAI-compatible clients.
3. Add capability negotiation so PRA can identify endpoint/backend and supported E0/E1/E2/E3 level.
4. Add E1 logical resource/version/session/task/delta transport only if it materially helps and can remain optional.
5. Reuse backend-native E2 where the current backend supports PRA. Propagate enough resource identity to prevent identical visible prompts with different PRA memories from colliding.
6. AUTO must safely downgrade to E0 when native support is absent.
7. Test model keep-alive, unload/reload, model switching, quantized variants and fingerprint invalidation.
8. Test multiple conversations, shared/independent documents, reconnects and gateway/model restarts.
9. Make onboarding CLI-first using the actual current `pra` CLI. Do not invent Ollama-only commands if generic runtime/gateway configuration suffices.
10. Run FULL, E0_SELECTED, E1 where relevant and E2 where inherited from the backend.
11. Measure setup complexity/time-to-first-PRA because Ollama's core product value is developer accessibility.

### Required metrics
Task quality, visible tokens, wire bytes, TTFT/ITL/completion p50/p95/p99, req/s, successful req/s, memory where exposed, model load/unload time, resource retransmission/reuse, cache hits where available, setup commands/time.

### Required tables
Ollama/backend mapping; deployment modes; model/backend/PRA compatibility; FULL/E0/E2 qualification; session/restart behavior; onboarding/usability.

### Figures
Gateway-to-native architecture; capability-negotiation flow; context-pressure curve; multi-session reuse; time-to-first-PRA workflow.

### Editorial structure
Introduction; PRA primer; Ollama/local-model UX; gateway-first integration; capability negotiation; native backend reuse; model/session lifecycle; evaluation; related work; limitations; reproducibility.

### Stop gate
E0 works with normal Ollama clients; backend architecture is verified; AUTO fallback is reliable; E1 only if useful; E2 inherited rather than duplicated where possible; reload/switch invalidation works; session/reconnect tests pass; CLI onboarding is documented; product matrix emitted; tests/PDF pass.

### Core message
Ollama is the local-model distribution and UX layer. PRA should integrate transparently above it and reuse deeper backend support when available, letting the same application move from E0 to E2 without becoming dependent on a PRA-specific Ollama fork.
