# Gemma 3 1B compute gate

## Architecture audit

- Checkpoint: `google/gemma-3-1b-it` (`dcc83ea841ab6100d6b47a070329e1ba4cf78752`)
- Decoder layers: 26
- Native local window: 512
- Exact native-global/PRA slots: (5, 11, 17, 23)
- Local layers remain unchanged.

## Local device

- Device: NVIDIA GeForce GTX 950M
- Python: 3.10.11
- PyTorch: 2.12.1+cu126
- CUDA available: True

The 1B checkpoint contains approximately 999,885,955 parameters.
Even a lower-bound mixed-precision Adam accounting (FP16 weights and gradients,
FP32 moments) is 11.2 GiB before activations, temporary
buffers, reference K/V, and checkpoint staging. The local 4 GiB GPU therefore
cannot provide the required 100--500-step full-weight benchmark safely.

## Decision

Do not launch G2--G5 training in this checkpoint. Configure distributed or
larger-memory training first, then run 100--500 measured steps and record
tokens/s, peak device memory, optimizer state, forward/backward time, dataloader
fraction, and checkpoint size. Extrapolate each full schedule before approval.
