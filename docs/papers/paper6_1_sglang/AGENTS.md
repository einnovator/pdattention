# Paper 6.1 execution contract

Paper 6.1 separates SGLang's exact prefix-radix address from PRA's scoped
resource/version/interval address. Arbitrary PRA resources must not be encoded
as fake radix prefixes.

Current status: patched SGLang-MLX environment measured with UnifiedRadixCache;
selected-text and native-K/V paths measured; generic logical-object control
plane implemented; local L1/L2/L3 HiCache, combined Radix-plus-native requests,
and matched original-answer QA measured. Distributed HiCache and queued server
tails remain open.

Any later native claim must preserve model topology, source positions, one
correct attention normalization, request lifetime, and authorization while
showing value beyond RadixAttention plus the black-box baseline.
