# Common Experiment Infrastructure

`common` contains reusable infrastructure for Transformer research experiments. It has
no dependency on a model implementation, retrieval system, or dataset package.

Use these modules directly in new experiments:

- `common.config`: base `TrainConfig`, YAML loading, and recursive config merging;
- `common.train`: training state, checkpoint lifecycle, device/batch helpers, and the
  callback-driven optimizer loop;
- `common.metrics`: running averages, perplexity, gradient norms, throughput, and CUDA
  allocation metrics;
- `common.logging`: console, TensorBoard, Weights & Biases, ClearML, and composite loggers;
- `common.plots`: JSON/Markdown metric histories and optional matplotlib plots;
- `common.callbacks` and `common.checkpointing`: early stopping, checkpoint paths, and
  model-agnostic restore helpers.

Architecture-specific packages should inject `batch_step`, `eval_step`, and optional
checkpoint metadata into `common.train.train_model`. They may subclass `TrainConfig` to
add their own service or model settings without adding those dependencies to `common`.
