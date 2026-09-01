# PRA routing adapter

- Base model: `mlx-community/Llama-3.1-8B-Instruct-4bit`
- Family: `llama`
- Routing representation: `attention_input_hidden_state_after_native_norm`
- Architecture: `asymmetric_linear` (4096 -> 128)
- Training data: QASPER, HOTPOTQA
- Parameters: 1,048,576

This artifact contains routing weights only, not base-model weights. See `config.json` for metrics, revisions, and reproducibility metadata.
