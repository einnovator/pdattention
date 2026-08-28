# Paper 6.2 execution contract

Paper 6.2 treats Apple unified memory, MLX prompt caching, rotating K/V, and
PRA non-prefix memory as distinct mechanisms. CUDA H2D language must not be
used for MLX measurements unless an actual discrete transfer exists.

Current status: MLX-LM server and direct Python cache experiments measured;
selected-text prompt-cache coexistence validated; rotating-cache archive
control measured over five seeds; native selected-K/V executor not implemented.

The mlx-lm HTTP server is an experimental deployment baseline and must retain
its upstream basic-security warning in product conclusions.

