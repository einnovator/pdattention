from dataclasses import dataclass, field
import warnings


@dataclass
class PRAConfig:
    """Architecture, routing, cache, and legacy training settings for PRA.

    The model first encodes each reference into layer-specific token K/V and one
    routing gist per chunk. During a normal forward pass, ``search_strategy``
    selects references/chunks and ``detail_materialization`` chooses which K/V
    is exposed to :class:`PRAttention`.
    """

    # Decoder dimensions. Tensors entering a block have shape [B, T, d_model].
    vocab_size: int = 128  # Tokenizer vocabulary and output-logit width.
    d_model: int = 128  # Hidden-state and routing-gist width.
    n_heads: int = 4  # Attention heads; each head has d_model / n_heads features.
    n_layers: int = 4  # Total decoder blocks.
    n_vanilla_layers: int = 0  # Leading blocks with only built-in self-attention.
    n_mixed_layers: int = 0  # Following blocks with self-attention then PRA.
    d_ff: int | None = None  # MLP width; defaults to 4 * d_model.
    max_seq_len: int = 128  # Largest prompt or independently encoded reference chunk.
    dropout: float = 0.0  # Dropout used by vanilla/mixed blocks and the GRU pooler.
    model_variant: str = "custom"  # custom, td_sa, td_pra, or last-two-layer tdx_pra.

    # Routing budgets and fusion. Remaining blocks after vanilla/mixed are PRA blocks.
    pra_layer_ids: tuple[int, ...] = (2, 3)  # Experiment metadata; variants normalize it.
    top_k_references: int = 2  # Maximum distinct URIs selected for each batch item/layer.
    top_k_chunks_per_reference: int = 1  # Maximum chunks retained from each selected URI.
    top_k_refs: int | None = None  # Deprecated alias for top_k_references.
    trigger_threshold: float = 0.2  # Drop selected chunks whose cosine score is lower.
    use_cross_attention_memory: bool = True  # Legacy flag; standalone PRA cross-attends.
    use_concat_memory: bool = False  # Reserved compatibility flag; concat is not implemented.
    memory_alpha: float = 0.5  # Scale in local_output + alpha * memory_output.

    # Search modes: hierarchical scores chunks then URIs; reference_first builds a
    # URI vector first; global_chunks ranks all chunks while enforcing both budgets.
    search_strategy: str = "hierarchical"
    reference_score_aggregation: str = "max"  # max, mean, or logsumexp over chunk scores.
    reference_level_gist_mode: str | None = None  # mean/last URI vector for reference_first.
    gist_mode: str = "mean"  # Pool chunk token keys by mean, last, ref_end, or GRU.
    max_gists_per_reference: int = 4  # Maximum independently routable chunks per URI.
    gist_overflow_policy: str = "truncate"  # truncate, merge_tail, or error.
    gist_gru_hidden_size: int | None = None  # GRU gist state width; defaults to d_model.
    gist_gru_num_layers: int = 1  # Recurrent depth for experimental GRU pooling.
    gist_gru_bidirectional: bool = False  # Read chunk keys in both directions when true.
    ref_end_token: str = "<REF_END>"  # Atomic marker selected by ref_end gist mode.

    # Reference partitioning and the detail made visible after routing.
    chunking_mode: str = "none"  # none, fixed token windows, markers, or plugin semantic.
    fixed_chunk_tokens: int = 64  # Window length used by fixed chunking.
    fixed_chunk_overlap_tokens: int = 0  # Repeated tokens between adjacent fixed windows.
    reference_overflow_policy: str = "truncate"  # Handling for a chunk over max_seq_len.
    marker_rules: tuple[str, ...] = ("<PRA_CHUNK>",)  # Explicit text split markers.
    semantic_chunker: object | None = None  # Plugin implementing SemanticChunker.
    detail_materialization: str = "selected_chunks"  # selected_chunks, full_reference, gist_only.

    # Variable memory lengths are padded per bucket before batched cross-attention.
    memory_bucket_count: int = 1  # Zero isolates items; positive values cap bucket count.
    memory_bucket_strategy: str = "optimal_contiguous"  # optimal_contiguous or equal_count.

    # Cache construction can be offline/detached or preserve gradients into gist creation.
    cache_build_mode: str = "detached"  # detached or trainable_gist.
    use_summary: bool = False  # Include a separately encoded summary in routing.
    summary_mode: str = "replace"  # replace, hybrid candidate scores, or normalized augment.

    # Recursive references are resolved child-first under shared depth and size budgets.
    recursive_refs_enabled: bool = False  # Let parent cache encoding attend to cached children.
    recursive_max_depth: int = 2  # Maximum child edges followed from a root URI.
    recursive_max_total_references: int = 16  # Cache entries built in one root traversal.
    recursive_max_total_tokens: int = 2048  # Resolved source tokens in one traversal.
    recursive_max_children_per_reference: int = 8  # Child URIs followed from one document.
    recursive_cycle_policy: str = "skip"  # skip, error, or link_only for a cycle/re-entry.
    recursive_missing_ref_policy: str = "warn"  # skip, warn, or error for unresolved URIs.

    # Diagnostics affect observability only, not routing or attention results.
    collect_detailed_timing: bool = False  # Record routing/materialization/attention durations.
    collect_attention_metrics: bool = True  # Compatibility flag; aggregates are always retained.
    collect_per_head_metrics: bool = False  # Reserved for per-head diagnostics.
    chunk_match_mode: str = "exact_id"  # Ground-truth match by ID, overlap, or IoU threshold.
    chunk_iou_threshold: float = 0.5  # Minimum span IoU when chunk_match_mode uses IoU.

    # Legacy convenience fields used by simple entry points; TrainConfig owns full training.
    batch_size: int = 8
    lr: float = 3e-4
    steps: int = 500
    device: str = "cuda"

    def __post_init__(self) -> None:
        """Normalize aliases/variants and reject incompatible mode settings early."""
        # Normalize routing choices before validating dependent fields.
        if self.top_k_refs is not None:
            warnings.warn(
                "top_k_refs is deprecated; use top_k_references and "
                "top_k_chunks_per_reference.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.top_k_references = int(self.top_k_refs)
        self.top_k_references = int(self.top_k_references)
        self.top_k_chunks_per_reference = int(self.top_k_chunks_per_reference)
        if self.top_k_references < 0 or self.top_k_chunks_per_reference < 0:
            raise ValueError("PRA top-k values must be non-negative; zero selects nothing.")
        if self.search_strategy not in {"hierarchical", "reference_first", "global_chunks"}:
            raise ValueError(f"Unsupported search_strategy: {self.search_strategy}")
        if self.reference_score_aggregation not in {"max", "mean", "logsumexp"}:
            raise ValueError(
                f"Unsupported reference_score_aggregation: {self.reference_score_aggregation}"
            )
        if self.reference_level_gist_mode not in {None, "mean", "last", "gru"}:
            raise ValueError(
                f"Unsupported reference_level_gist_mode: {self.reference_level_gist_mode}"
            )
        if self.gist_mode not in {"mean", "last", "ref_end", "gru"}:
            raise ValueError(f"Unsupported gist_mode: {self.gist_mode}")
        self.max_gists_per_reference = int(self.max_gists_per_reference)
        if self.max_gists_per_reference <= 0:
            raise ValueError("max_gists_per_reference must be positive.")
        if self.gist_overflow_policy not in {"truncate", "merge_tail", "error"}:
            raise ValueError(f"Unsupported gist_overflow_policy: {self.gist_overflow_policy}")
        if self.chunking_mode not in {"none", "fixed", "markers", "semantic"}:
            raise ValueError(f"Unsupported chunking_mode: {self.chunking_mode}")
        self.fixed_chunk_tokens = int(self.fixed_chunk_tokens)
        self.fixed_chunk_overlap_tokens = int(self.fixed_chunk_overlap_tokens)
        if self.fixed_chunk_tokens <= 0:
            raise ValueError("fixed_chunk_tokens must be positive.")
        if not 0 <= self.fixed_chunk_overlap_tokens < self.fixed_chunk_tokens:
            raise ValueError(
                "fixed_chunk_overlap_tokens must be non-negative and smaller than fixed_chunk_tokens."
            )
        if self.reference_overflow_policy not in {"truncate", "error"}:
            raise ValueError(
                f"Unsupported reference_overflow_policy: {self.reference_overflow_policy}"
            )
        self.marker_rules = tuple(str(value) for value in self.marker_rules)
        if self.chunking_mode == "semantic" and self.semantic_chunker is None:
            raise NotImplementedError(
                "chunking_mode='semantic' requires an explicit semantic_chunker implementation."
            )
        if self.detail_materialization not in {"selected_chunks", "full_reference", "gist_only"}:
            raise ValueError(f"Unsupported detail_materialization: {self.detail_materialization}")
        self.memory_bucket_count = int(self.memory_bucket_count)
        if self.memory_bucket_count < 0:
            raise ValueError("memory_bucket_count must be non-negative.")
        if self.memory_bucket_strategy not in {"optimal_contiguous", "equal_count"}:
            raise ValueError(f"Unsupported memory_bucket_strategy: {self.memory_bucket_strategy}")
        if self.cache_build_mode not in {"detached", "trainable_gist"}:
            raise ValueError(f"Unsupported cache_build_mode: {self.cache_build_mode}")
        if self.summary_mode not in {"replace", "hybrid", "augment"}:
            raise ValueError(f"Unsupported summary_mode: {self.summary_mode}")
        if self.recursive_cycle_policy not in {"skip", "error", "link_only"}:
            raise ValueError(f"Unsupported recursive_cycle_policy: {self.recursive_cycle_policy}")
        if self.recursive_missing_ref_policy not in {"skip", "warn", "error"}:
            raise ValueError(
                f"Unsupported recursive_missing_ref_policy: {self.recursive_missing_ref_policy}"
            )
        # A single traversal shares these limits across every recursively built child.
        for name in (
            "recursive_max_depth",
            "recursive_max_total_references",
            "recursive_max_total_tokens",
            "recursive_max_children_per_reference",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")
            setattr(self, name, value)
        if self.chunk_match_mode not in {"exact_id", "any_overlap", "iou_threshold"}:
            raise ValueError(f"Unsupported chunk_match_mode: {self.chunk_match_mode}")
        if not 0.0 <= float(self.chunk_iou_threshold) <= 1.0:
            raise ValueError("chunk_iou_threshold must be between zero and one.")
        self.d_ff = 4 * self.d_model if self.d_ff is None else int(self.d_ff)
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive.")
        # Named variants map the paper's architecture labels onto block counts.
        variants = {"custom", "td_sa", "td_pra", "tdx_pra"}
        if self.model_variant not in variants:
            raise ValueError(f"Unsupported model_variant: {self.model_variant}")
        if self.model_variant == "td_sa":
            self.n_vanilla_layers = self.n_layers
            self.n_mixed_layers = 0
            self.pra_layer_ids = tuple()
        elif self.model_variant == "td_pra":
            self.n_vanilla_layers = 0
            self.n_mixed_layers = 0
            self.pra_layer_ids = tuple(range(self.n_layers))
        elif self.model_variant == "tdx_pra":
            self.n_vanilla_layers = max(self.n_layers - 2, 0)
            self.n_mixed_layers = 0
            self.pra_layer_ids = tuple(range(self.n_vanilla_layers, self.n_layers))
        self.n_vanilla_layers = int(self.n_vanilla_layers)
        self.n_mixed_layers = int(self.n_mixed_layers)
        if self.n_vanilla_layers < 0 or self.n_mixed_layers < 0:
            raise ValueError("n_vanilla_layers and n_mixed_layers must be non-negative.")
        if self.n_vanilla_layers + self.n_mixed_layers > self.n_layers:
            raise ValueError("n_vanilla_layers + n_mixed_layers cannot exceed n_layers.")


@dataclass
class ResolverServiceConfig:
    """Select the URI resolver backend used before cache construction."""

    type: str = "in_memory"  # Registered resolver implementation name.
    options: dict = field(default_factory=dict)  # Backend-specific constructor arguments.

    @classmethod
    def from_value(cls, value) -> "ResolverServiceConfig":
        """Normalize shorthand strings/dicts into a typed service configuration."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(type=value)
        if isinstance(value, dict):
            return cls(type=str(value.get("type", "in_memory")), options=dict(value.get("options") or {}))
        raise TypeError(f"Unsupported resolver service config: {value!r}")


@dataclass
class CacheServiceConfig:
    """Select the storage/routing backend that holds encoded reference memory."""

    type: str = "simple"  # Registered cache implementation name.
    options: dict = field(default_factory=dict)  # Backend-specific constructor arguments.

    @classmethod
    def from_value(cls, value) -> "CacheServiceConfig":
        """Normalize shorthand strings/dicts into a typed service configuration."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(type=value)
        if isinstance(value, dict):
            return cls(type=str(value.get("type", "simple")), options=dict(value.get("options") or {}))
        raise TypeError(f"Unsupported cache service config: {value!r}")


@dataclass
class TrainConfig:
    """Generic loop, data, logging, and service settings for a PRA experiment."""

    experiment_name: str = "standalone_tiny"  # Run-directory and logger label.
    output_dir: str = "out"  # Parent directory for checkpoints, metrics, and traces.
    seed: int = 0  # Shared Python, NumPy, and PyTorch random seed.
    device: str = "auto"  # auto, cpu, cuda, or another PyTorch device string.
    dtype: str = "float32"  # Requested parameter/compute dtype.
    epochs: int = 3  # Maximum complete passes over the training loader.
    max_steps: int | None = None  # Optional optimizer-step cap across epochs.
    batch_size: int = 8  # Samples collated per dataloader batch.
    grad_accum_steps: int = 1  # Backward passes accumulated per optimizer update.
    learning_rate: float = 3e-4  # Optimizer base learning rate.
    weight_decay: float = 0.0  # AdamW decoupled weight decay.
    warmup_steps: int = 0  # Linear scheduler warm-up optimizer steps.
    max_grad_norm: float = 1.0  # Global gradient-norm clipping threshold.
    eval_every_steps: int = 50  # Validation cadence in optimizer steps.
    save_every_steps: int = 100  # Latest-checkpoint cadence in optimizer steps.
    log_every_steps: int = 10  # Batch/optimizer metric logging cadence.
    num_workers: int = 0  # Worker processes used by each dataloader.
    pin_memory: bool = False  # Pin host batches for faster CUDA transfer.
    persistent_workers: bool = False  # Keep workers alive between epochs.
    resume_from: str | None = None  # Checkpoint path restored before training.
    use_tensorboard: bool = True  # Emit TensorBoard scalar/text events.
    save_metric_plots: bool = True  # Render metric-history plots at run close.
    use_wandb: bool = False  # Enable optional Weights & Biases logging.
    use_clearml: bool = False  # Enable optional ClearML logging.
    mixed_precision: bool = False  # Use CUDA autocast and gradient scaling.
    early_stopping_patience: int | None = None  # Validations without improvement before stop.
    dataset_stage: str = "stage0_synthetic_memory"  # Dataset directory/name to load.
    data_dir: str = "data"  # Root containing generated dataset stages.
    max_examples: int | None = None  # Optional dataset-size limit for quick runs.
    max_seq_len: int = 96  # Collator sequence length; should not exceed model capacity.
    shuffle: bool = True  # Shuffle the training split each epoch.
    resolver_config: ResolverServiceConfig = field(default_factory=ResolverServiceConfig)
    cache_config: CacheServiceConfig = field(default_factory=CacheServiceConfig)

    def __post_init__(self) -> None:
        """Normalize nested resolver/cache settings accepted from YAML or Python."""
        self.resolver_config = ResolverServiceConfig.from_value(self.resolver_config)
        self.cache_config = CacheServiceConfig.from_value(self.cache_config)
