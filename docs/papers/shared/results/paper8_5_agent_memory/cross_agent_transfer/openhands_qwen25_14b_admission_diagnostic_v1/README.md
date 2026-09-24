# OpenHands / Qwen2.5-Coder-14B admission diagnostic

This audit holds the model, revision, temperature, tokenizer and task workspace
fixed while changing only OpenHands' tool protocol and exposed tool surface.
Every attempt is a **FULL** control: no history record was selected out, so none
of the failures can be attributed to PRA selection.

Provider-native tool calling failed before the first action because the MLX
OpenAI-compatible endpoint returned tool examples as ordinary assistant text.
The explicit prompt-mocked fallback repaired that protocol boundary.  With the
default OpenHands terminal/editor/tracker surface on Task 1, however, the model
searched for `fields.py`, selected the wrong file, and repeated the same invalid
editor replacement five times despite five explicit error observations.

A transverse terminal-only surface removed that editor loop and reduced the
initial prompt from 7,739 to 5,578 tokens, but did not admit Task 1.  The model
located the correct file, then used a global `sed` substitution that corrupted
both `IntegerField` and `BinaryField`; two failed verification calls led to
further corruptive substitutions.  On Task 4 the same shell surface localized
`get_format()` but invented an `allow_lazy` parameter rather than the required
`str(format_type)` repair, then moved into environment setup after failed
verification.

The result is a scaffold/model admission failure, not evidence against the
history policy.  Mini-swe-agent has a successful Task-1 control with this exact
Qwen2.5-Coder-14B revision, while OpenHands does not yet have a successful
paired identity.  Therefore no OpenHands selective-history arm is launched for
this model.  The already qualified OpenHands/Qwen3-Coder-30B cohort remains the
valid cross-agent policy-transfer evidence.

Operationally, stock MLX prompt-cache reuse also deep-copied a 1.96 GB cache and
forced the 16 GB host into swap.  The final diagnostics disabled that ordinary
prompt cache and bounded MLX's freed-buffer cache/wired allocation.  Those
controls affect execution feasibility only and are not reported as PRA savings.

The economical action traces and remote immutable-source hashes are recorded in
`audit.json`.  Raw event histories remain in the immutable run directories
listed there; they include the full OpenHands system prompt and are intentionally
not duplicated into the paper bundle.
