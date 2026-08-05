from dataclasses import dataclass, field
import warnings


@dataclass
class PRAConfig:
    """Model and PRA architecture hyperparameters."""

    vocab_size: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    n_vanilla_layers: int = 0
    n_mixed_layers: int = 0
    d_ff: int | None = None
    max_seq_len: int = 128
    dropout: float = 0.0
    model_variant: str = "custom"

    # PRA-specific
    pra_layer_ids: tuple[int, ...] = (2, 3)
    top_k_references: int = 2
    top_k_chunks_per_reference: int = 1
    top_k_refs: int | None = None
    trigger_threshold: float = 0.2
    use_cross_attention_memory: bool = True
    use_concat_memory: bool = False
    memory_alpha: float = 0.5
    search_strategy: str = "hierarchical"
    reference_score_aggregation: str = "max"
    reference_level_gist_mode: str | None = None
    gist_mode: str = "mean"
    max_gists_per_reference: int = 4
    gist_overflow_policy: str = "truncate"
    gist_gru_hidden_size: int | None = None
    gist_gru_num_layers: int = 1
    gist_gru_bidirectional: bool = False
    ref_end_token: str = "<REF_END>"
    chunking_mode: str = "none"
    fixed_chunk_tokens: int = 64
    fixed_chunk_overlap_tokens: int = 0
    reference_overflow_policy: str = "truncate"
    marker_rules: tuple[str, ...] = ("<PRA_CHUNK>",)
    semantic_chunker: object | None = None
    detail_materialization: str = "selected_chunks"
    memory_bucket_count: int = 1
    memory_bucket_strategy: str = "optimal_contiguous"
    cache_build_mode: str = "detached"
    use_summary: bool = False
    summary_mode: str = "replace"
    recursive_refs_enabled: bool = False
    recursive_max_depth: int = 2
    recursive_max_total_references: int = 16
    recursive_max_total_tokens: int = 2048
    recursive_max_children_per_reference: int = 8
    recursive_cycle_policy: str = "skip"
    recursive_missing_ref_policy: str = "warn"
    collect_detailed_timing: bool = False
    collect_attention_metrics: bool = True
    collect_per_head_metrics: bool = False
    chunk_match_mode: str = "exact_id"
    chunk_iou_threshold: float = 0.5

    # Training
    batch_size: int = 8
    lr: float = 3e-4
    steps: int = 500
    device: str = "cuda"

    def __post_init__(self) -> None:
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
    """Configuration for constructing a reference resolver service."""

    type: str = "in_memory"
    options: dict = field(default_factory=dict)

    @classmethod
    def from_value(cls, value) -> "ResolverServiceConfig":
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
    """Configuration for constructing a PRA memory cache service."""

    type: str = "simple"
    options: dict = field(default_factory=dict)

    @classmethod
    def from_value(cls, value) -> "CacheServiceConfig":
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
    """Standalone trainer, dataloader, logging, and checkpoint settings."""

    experiment_name: str = "standalone_tiny"
    output_dir: str = "out"
    seed: int = 0
    device: str = "auto"
    dtype: str = "float32"
    epochs: int = 3
    max_steps: int | None = None
    batch_size: int = 8
    grad_accum_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    eval_every_steps: int = 50
    save_every_steps: int = 100
    log_every_steps: int = 10
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    resume_from: str | None = None
    use_tensorboard: bool = True
    save_metric_plots: bool = True
    use_wandb: bool = False
    use_clearml: bool = False
    mixed_precision: bool = False
    early_stopping_patience: int | None = None
    dataset_stage: str = "stage0_synthetic_memory"
    data_dir: str = "data"
    max_examples: int | None = None
    max_seq_len: int = 96
    shuffle: bool = True
    resolver_config: ResolverServiceConfig = field(default_factory=ResolverServiceConfig)
    cache_config: CacheServiceConfig = field(default_factory=CacheServiceConfig)

    def __post_init__(self) -> None:
        self.resolver_config = ResolverServiceConfig.from_value(self.resolver_config)
        self.cache_config = CacheServiceConfig.from_value(self.cache_config)
