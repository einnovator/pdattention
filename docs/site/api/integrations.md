# Hugging Face Adapter

The Hugging Face wrapper is an experimental compatibility path. It adds a PRA-style
memory branch to selected decoder layers without replacing the pretrained model's native
self-attention implementation.

::: hf_wrappers.pra_wrapper

## Unified Runtime

`PRARuntime` combines the Hugging Face adapter with physical K/V planning,
lazy capability records, safe tool execution, and session-scoped compact
result backing.

::: pra_hf.runtime

## Tools and Skills

::: pra_hf.capability_sdk

::: pra_hf.capability_runtime

::: pra_hf.tool_records

::: pra_hf.skill_records

## Compact Results

::: pra_hf.typed_context

::: pra_hf.adaptive_context_runtime

::: pra_hf.progressive_context
