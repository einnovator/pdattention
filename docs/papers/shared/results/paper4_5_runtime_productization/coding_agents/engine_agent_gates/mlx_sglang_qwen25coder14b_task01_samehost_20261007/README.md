# Same-host direct-MLX versus SGLang-MLX autonomous gate

This gate holds host, checkpoint, tokenizer, task image, mini-swe-agent
scaffold, sampling controls and model-visible observation stream fixed. Plain,
PRA-100 and matched tail-90 resolve on both engines in six calls. The two
engines produce identical assistant-content hashes within each arm, identical
six-command trajectories, and the same patch.

The logical ledger is also identical. Tail-90 materializes 14,075 of 14,731
message-content tokens and selects 14,225 K/V tokens on both engines. This is
4.45% own materialized saving, 4.09% paired input-plus-completion saving, and
4.71% selected-K/V reduction relative to PRA-100. The short one-task trajectory
does not provide the 30--50% opportunity expected in persistent sessions.

Physical realization is not identical. Both engines re-encode zero selected
history tokens and copy zero selected K/V for attachment. Direct MLX reports
8.36 GB of total K/V movement and a 1.23 GB temporary peak for tail-90;
SGLang-MLX reports 37.16 MB and 1.61 GB respectively. Direct MLX passes its
packed identical-subset reference with maximum raw-logit delta 0.0 on all three
sparse requests.

An otherwise similar plain run on the M5 host is excluded. Its first recursive
`grep` observation had a different line order, producing a malformed edit and
a failed official grade. Repeating direct MLX on the same M4 host as SGLang
restored the complete six-action trajectory. That row is an observation-order
control, not an engine result.

Full artifact locations and SHA-256 identities are recorded in `summary.json`.
