# Hugging Face Adapter

The Hugging Face wrapper is an experimental compatibility path. It adds a PRA-style
memory branch to selected decoder layers without replacing the pretrained model's native
self-attention implementation.

::: hf_wrappers.pra_wrapper

## Typed Capability SDK

The provider-neutral capability facade normalizes Python callables, explicit
`Skill` objects, and OpenAI- or Anthropic-style skill folders. Compact selection
views and complete schemas/instructions are encoded lazily by default.

::: pra_hf.capability_sdk

::: pra_hf.capability_runtime

::: pra_hf.skill_records

::: pra_hf.context_records
