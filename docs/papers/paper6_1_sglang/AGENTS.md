# Paper 6.1 execution contract

Paper 6.1 separates SGLang's exact prefix-radix address from PRA's scoped
resource/version/interval address. Arbitrary PRA resources must not be encoded
as fake radix prefixes.

Current status: patched SGLang-MLX environment measured with UnifiedRadixCache;
selected-text black-box baseline measured; generic logical-object control plane
implemented; HiCache and selected native PRA K/V not measured.

Any later native claim must preserve model topology, source positions, one
correct attention normalization, request lifetime, and authorization while
showing value beyond RadixAttention plus the black-box baseline.

