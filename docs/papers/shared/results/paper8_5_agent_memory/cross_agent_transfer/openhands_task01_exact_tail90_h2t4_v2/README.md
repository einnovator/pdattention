# OpenHands Task 1 exact-tokenizer tail-90 qualification

This pair corrects the historical transfer proxy's whitespace-token diagnostic.
Both arms use OpenHands SDK 1.49.2, `qwen3-coder:30b` at observed digest
`06c1097e...90bca`, its exact tokenizer revision, temperature zero, disabled
native condensation, the same workspace image, and the ordinary OpenAI-tools
protocol.  The selective arm uses the same matched token-tail parameters as
the new mini-swe-agent bridge: protected head 2, protected tail 4, and a
nominal 90% hard ceiling.

Both patches pass the official SWE-bench grader and have the identical
689-byte SHA-256 `8512d6bc...44aae`.  FULL takes 33 actions and 509,381 exact
message-content tokens.  Tail-90 takes 24 actions, exposes 407,493 candidate-own
full-history tokens, and materializes 187,138.  This is 54.08% own logical
saving and 63.26% paired input saving; provider prompt tokens fall 54.97%.

The ceiling is not a retention target.  Whole causal groups leave budget
unused: 19 of 24 requests underfill their ceiling, logical retention ranges
from 29.13% to 100%, and the final request retains 45.83%.  The old
whitespace-token figures are not pooled with this result.

`full/` and `tail90/` contain native events, exact-token request traces,
manifests, patches, proxy identities, and independent official reports.
`qualification.json` is the compact paired reduction.
