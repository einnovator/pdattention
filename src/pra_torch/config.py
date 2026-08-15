from dataclasses import dataclass, field
import warnings

from common.config import TrainConfig as CommonTrainConfig
from .chunking import ChunkingConfig


@dataclass
class PRAConfig:
    """Architecture, routing, cache, and legacy training settings for PRA.

    The model first encodes each reference into layer-specific token K/V and
    configurable routing-gist sets per chunk and URI. During a normal forward pass, ``search_strategy``
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
    model_max_context_tokens: int | None = None  # Hard native-operation context ceiling.
    position_encoding: str = "absolute"  # absolute (default), rope, or sinusoidal.
    rope_theta: float = 10_000.0  # RoPE base controlling angular frequency by head feature.
    self_attention_window: int | None = None  # None is global; finite W includes current token.
    dropout: float = 0.0  # Dropout used by vanilla/mixed blocks and the GRU pooler.
    model_variant: str = "custom"  # custom, td_sa, td_pra, tdx_pra, or td_layered_pra.

    # Routing budgets and transport. Remaining blocks after vanilla/mixed are PRA blocks.
    pra_layer_ids: tuple[int, ...] = (2, 3)  # Experiment metadata; variants normalize it.
    top_k_references: int = 2  # Maximum distinct URIs selected for each batch item/layer.
    top_k_chunks_per_reference: int = 1  # Maximum chunks retained from each selected URI.
    top_k_refs: int | None = None  # Deprecated alias for top_k_references.
    trigger_threshold: float = 0.2  # Drop selected chunks whose cosine score is lower.
    memory_transport: str = "native_kv"  # native_kv (canonical) or adapted cross_attention.
    use_cross_attention_memory: bool | None = None  # Deprecated checkpoint/config alias.
    use_concat_memory: bool | None = None  # Deprecated alias for native_kv transport.
    memory_alpha: float = 0.5  # Cross-attention-only residual scale; ignored by native_kv.

    # Long prompts keep a bounded recent tail and can expose displaced history as PRA memory.
    max_prompt_direct_tokens: int | None = None  # None inherits max_seq_len.
    prompt_overflow_mode: str = "truncate"  # truncate, implicit_reference, or error.
    max_prompt_gists: int | None = None  # Prompt-head chunk cap; None keeps every chunk.
    max_materialized_memory_tokens: int | None = None  # Deployment cap after routing.
    context_safety_reserve_tokens: int = 0  # Native positions reserved beyond direct+memory.

    # Search modes and the independent chunk/reference gist representations.
    search_strategy: str = "hierarchical"
    routing_backend: str = "tensorized"  # tensorized exact search or legacy scalar reference path.
    reference_score_aggregation: str = "max"  # max, mean, or logsumexp over chunk scores.
    reference_level_gist_mode: str | None = None  # URI gist strategy for reference_first.
    reference_gists_per_reference: int = 1  # Requested URI-level gists per layer.
    reference_gist_score_aggregation: str = "max"  # Reduce query scores over URI gists.
    gist_mode: str = "mean"  # Chunk gist strategy; single, segmented, or multi-prototype.
    gists_per_chunk: int = 1  # Requested chunk-level gists; single modes still produce one.
    gist_score_aggregation: str = "max"  # Reduce query scores over one chunk's gist set.
    max_gists_per_reference: int = 4  # Maximum independently routable chunks per URI.
    gist_overflow_policy: str = "truncate"  # truncate, merge_tail, or error.
    gist_gru_hidden_size: int | None = None  # GRU gist state width; defaults to d_model.
    gist_gru_num_layers: int = 1  # Recurrent depth for experimental GRU pooling.
    gist_gru_bidirectional: bool = False  # Read chunk keys in both directions when true.
    ref_end_token: str = "<REF_END>"  # Atomic marker selected by ref_end gist mode.

    # Multi-gist strategies cluster keys and aggregate values with the same assignments.
    gist_kmeans_max_iters: int = 8  # Maximum local Lloyd iterations.
    gist_kmeans_init: str = "kmeans++"  # kmeans++ or deterministic seeded sample.
    gist_kmeans_tol: float = 1e-4  # Stop when maximum centroid movement is below this value.
    gist_kmeans_normalize: bool = True  # Cluster normalized projected keys.
    gist_kmeans_seed: int = 0  # Local RNG seed; global torch state is never consumed.
    gist_kmeans_empty_cluster_policy: str = "farthest"  # farthest reseed or error.
    gist_som_steps: int = 32  # Local self-organizing-map update count.
    gist_som_learning_rate: float = 0.2  # Initial SOM prototype learning rate.
    gist_som_final_learning_rate: float = 0.05  # Final SOM learning rate.
    gist_som_neighborhood_radius: float = 1.0  # Initial line-topology radius.
    gist_som_final_neighborhood_radius: float = 0.0  # Final line-topology radius.
    gist_som_distance: str = "cosine"  # cosine or euclidean winner selection.
    gist_som_normalize: bool = True  # Keep SOM input/prototypes in normalized key space.
    gist_som_init: str = "sample"  # Deterministic seeded sample initialization.
    gist_som_seed: int = 0  # Local SOM RNG seed.
    gist_som_topology: str = "line"  # Supported local prototype topology.
    gist_prototype_method: str = "farthest"  # Diversity-selection method.
    gist_prototype_init: str = "mean_nearest"  # mean_nearest or deterministic sample.
    gist_prototype_refine: bool = True  # Replace selected points with assigned K means.
    gist_prototype_normalize: bool = True  # Select prototypes in normalized key space.
    gist_prototype_distance: str = "cosine"  # cosine or euclidean assignments.
    gist_prototype_seed: int = 0  # Local prototype RNG seed for sample initialization.
    gist_hybrid_global_mode: str = "mean"  # Single gist prepended to local prototypes.
    gist_hybrid_local_mode: str = "kmeans"  # kmeans, som, or prototype.
    gist_hybrid_global_count: int = 1  # Requested global slots; single modes yield one.
    gist_hybrid_deduplicate: bool = True  # Remove nearly identical global/local gists.
    gist_hybrid_min_cosine_separation: float = 0.0  # Minimum retained pairwise distance.

    # Reference partitioning and the detail made visible after routing.
    chunking_mode: str = "none"  # none, fixed token windows, markers, or plugin semantic.
    fixed_chunk_tokens: int = 64  # Window length used by fixed chunking.
    fixed_chunk_overlap_tokens: int = 0  # Repeated tokens between adjacent fixed windows.
    chunk_overlap_fraction: float = 0.0  # Alternative fractional fixed-window overlap.
    overlap_materialization: str = "deduplicate"  # deduplicate or keep_duplicates.
    reference_encoding_strategy: str = "independent"  # independent, block_slice, native_slice.
    encoding_block_references: int = 8  # Consecutive URIs contextualized per block_slice.
    encoding_overlap_fraction: float = 0.0  # Left-context duplication during block encoding.
    reference_position_mode: str = "local"  # Reset each block or preserve global offsets.
    prompt_position_mode: str = "local"  # local or continue after historical source K/V.
    reference_overflow_policy: str = "truncate"  # Handling for a chunk over max_seq_len.
    kv_cache_residency: str = "gpu"  # Store full native token K/V on gpu or cpu.
    kv_cache_pin_memory: bool = False  # Page-lock CPU K/V for faster host-to-device copies.
    kv_cache_non_blocking: bool = False  # Request asynchronous selected-K/V transfers.
    marker_rules: tuple[str, ...] = ("<PRA_CHUNK>",)  # Explicit text split markers.
    semantic_chunker: object | None = None  # Plugin implementing SemanticChunker.
    encoding_chunking: ChunkingConfig | dict | None = None  # Model-safe contextual blocks.
    routing_chunking: ChunkingConfig | dict | None = None  # Smaller addressable K/V slices.
    encoding_context_mode: str = "independent"  # independent or overlap/historical_window.
    # Detail disclosed after conceptual routing. Paper 3 interval modes consume
    # source-relative ``materialization_intervals`` attached to selected hits.
    detail_materialization: str = "selected_chunks"

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
    collect_routing_metrics: bool = False  # Keep complete rankings for aggregate MRR/recall.
    collect_rank_diagnostics: bool = False  # Retain complete pre-top-k candidate score lists.
    chunk_match_mode: str = "exact_id"  # Ground-truth match by ID, overlap, or IoU threshold.
    chunk_iou_threshold: float = 0.5  # Minimum span IoU when chunk_match_mode uses IoU.

    # Legacy convenience fields used by simple entry points; TrainConfig owns full training.
    batch_size: int = 8
    lr: float = 3e-4
    steps: int = 500
    device: str = "cuda"

    @property
    def effective_prompt_direct_tokens(self) -> int:
        """Return direct-token capacity after the hard model reserve."""
        requested = self.max_prompt_direct_tokens or self.effective_model_max_context_tokens
        available = self.effective_model_max_context_tokens - self.context_safety_reserve_tokens
        return min(int(requested), int(self.max_seq_len), int(available))

    @property
    def effective_model_max_context_tokens(self) -> int:
        """Return the hard native-operation limit, defaulting to model capacity."""
        return int(self.model_max_context_tokens or self.max_seq_len)

    @property
    def position_capacity(self) -> int | None:
        """Return the finite position-ID range, or None for mechanically unbounded modes."""
        return self.max_seq_len if self.position_encoding == "absolute" else None

    @property
    def routing_chunking_config(self) -> ChunkingConfig:
        """Return explicit routing policy or migrate legacy chunking fields."""
        if self.routing_chunking is not None:
            return ChunkingConfig.from_value(self.routing_chunking)
        return ChunkingConfig(
            mode=self.chunking_mode,
            chunk_tokens=self.fixed_chunk_tokens,
            overlap_fraction=self.chunk_overlap_fraction,
            overlap_tokens=self.fixed_chunk_overlap_tokens,
            markers=tuple(self.marker_rules),
            semantic_chunker=self.semantic_chunker,
        )

    @property
    def encoding_chunking_config(self) -> ChunkingConfig:
        """Return the model-call partition policy; ``none`` means auto-bound only."""
        if self.encoding_chunking is not None:
            return ChunkingConfig.from_value(self.encoding_chunking)
        return ChunkingConfig(
            mode="none",
            chunk_tokens=None,
            overlap_fraction=self.encoding_overlap_fraction,
        )

    @property
    def resolved_chunk_overlap_tokens(self) -> int:
        """Resolve the mutually exclusive token/fraction overlap configuration."""
        return self.routing_chunking_config.resolved_overlap_tokens(
            self.routing_chunking_config.chunk_tokens
        )

    def prompt_tail_position_offset(self, head_tokens: int, direct_tokens: int) -> int:
        """Continue source coordinates while IDs fit the positional mechanism."""
        if (
            self.prompt_position_mode == "historical"
            and (
                self.position_encoding in {"rope", "sinusoidal"}
                or int(head_tokens) + int(direct_tokens)
                <= self.max_seq_len
            )
        ):
            return int(head_tokens)
        return 0

    def __post_init__(self) -> None:
        """Normalize aliases/variants and reject incompatible mode settings early."""
        self.position_encoding = str(self.position_encoding).lower()
        if self.position_encoding not in {"absolute", "rope", "sinusoidal"}:
            raise ValueError(f"Unsupported position_encoding: {self.position_encoding}")
        self.rope_theta = float(self.rope_theta)
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive.")
        if self.self_attention_window is not None:
            self.self_attention_window = int(self.self_attention_window)
            if self.self_attention_window <= 0:
                raise ValueError("self_attention_window must be positive or None.")
        if self.position_encoding == "rope" and (self.d_model // self.n_heads) % 2:
            raise ValueError("RoPE requires an even d_model / n_heads head dimension.")
        self.max_seq_len = int(self.max_seq_len)
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")
        if self.model_max_context_tokens is not None:
            self.model_max_context_tokens = int(self.model_max_context_tokens)
            if self.model_max_context_tokens <= 0:
                raise ValueError("model_max_context_tokens must be positive or None.")
            if self.model_max_context_tokens > self.max_seq_len:
                raise ValueError(
                    "model_max_context_tokens cannot exceed this model's max_seq_len."
                )
        self.context_safety_reserve_tokens = int(self.context_safety_reserve_tokens)
        if not 0 <= self.context_safety_reserve_tokens < self.effective_model_max_context_tokens:
            raise ValueError("context_safety_reserve_tokens must fit inside the model limit.")
        if self.max_materialized_memory_tokens is not None:
            self.max_materialized_memory_tokens = int(self.max_materialized_memory_tokens)
            if self.max_materialized_memory_tokens <= 0:
                raise ValueError("max_materialized_memory_tokens must be positive or None.")
        # Normalize routing choices before validating dependent fields.
        if self.use_cross_attention_memory is not None:
            warnings.warn(
                "use_cross_attention_memory is deprecated; use memory_transport.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.memory_transport = (
                "cross_attention" if self.use_cross_attention_memory else "native_kv"
            )
        if self.use_concat_memory:
            warnings.warn(
                "use_concat_memory is deprecated; native_kv is the concatenated-K/V mode.",
                DeprecationWarning,
                stacklevel=2,
            )
            if self.memory_transport == "cross_attention":
                raise ValueError("Legacy cross-attention and concat-memory flags conflict.")
            self.memory_transport = "native_kv"
        if self.memory_transport not in {"native_kv", "cross_attention"}:
            raise ValueError(f"Unsupported memory_transport: {self.memory_transport}")
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
        if self.max_prompt_direct_tokens is not None:
            self.max_prompt_direct_tokens = int(self.max_prompt_direct_tokens)
            if self.max_prompt_direct_tokens <= 0:
                raise ValueError("max_prompt_direct_tokens must be positive or None.")
        if self.prompt_overflow_mode not in {"truncate", "implicit_reference", "error"}:
            raise ValueError(f"Unsupported prompt_overflow_mode: {self.prompt_overflow_mode}")
        if self.max_prompt_gists is not None:
            self.max_prompt_gists = int(self.max_prompt_gists)
            if self.max_prompt_gists <= 0:
                raise ValueError("max_prompt_gists must be positive or None.")
        if self.effective_prompt_direct_tokens <= 0:
            raise ValueError("The model context reserve leaves no direct prompt capacity.")
        if self.search_strategy not in {"hierarchical", "reference_first", "global_chunks"}:
            raise ValueError(f"Unsupported search_strategy: {self.search_strategy}")
        if self.routing_backend not in {"tensorized", "legacy"}:
            raise ValueError(f"Unsupported routing_backend: {self.routing_backend}")
        if self.reference_score_aggregation not in {"max", "mean", "logsumexp"}:
            raise ValueError(
                f"Unsupported reference_score_aggregation: {self.reference_score_aggregation}"
            )
        gist_modes = {
            "mean",
            "segment_mean",
            "last",
            "ref_end",
            "gru",
            "kmeans",
            "som",
            "prototype",
            "hybrid",
        }
        if self.reference_level_gist_mode not in {None, *gist_modes}:
            raise ValueError(
                f"Unsupported reference_level_gist_mode: {self.reference_level_gist_mode}"
            )
        if self.reference_level_gist_mode == "ref_end":
            raise ValueError("reference_level_gist_mode='ref_end' has no URI-level token marker.")
        if self.gist_mode not in gist_modes:
            raise ValueError(f"Unsupported gist_mode: {self.gist_mode}")
        self.gists_per_chunk = int(self.gists_per_chunk)
        self.reference_gists_per_reference = int(self.reference_gists_per_reference)
        if self.gists_per_chunk <= 0 or self.reference_gists_per_reference <= 0:
            raise ValueError("Chunk and reference gist counts must be positive.")
        gist_aggregations = {"max", "mean", "logsumexp"}
        if self.gist_score_aggregation not in gist_aggregations:
            raise ValueError(f"Unsupported gist_score_aggregation: {self.gist_score_aggregation}")
        if self.reference_gist_score_aggregation not in gist_aggregations:
            raise ValueError(
                "Unsupported reference_gist_score_aggregation: "
                f"{self.reference_gist_score_aggregation}"
            )
        self.max_gists_per_reference = int(self.max_gists_per_reference)
        if self.max_gists_per_reference <= 0:
            raise ValueError("max_gists_per_reference must be positive.")
        if self.gist_overflow_policy not in {"truncate", "merge_tail", "error"}:
            raise ValueError(f"Unsupported gist_overflow_policy: {self.gist_overflow_policy}")
        self.gist_kmeans_max_iters = int(self.gist_kmeans_max_iters)
        if self.gist_kmeans_max_iters <= 0 or float(self.gist_kmeans_tol) < 0:
            raise ValueError("K-means iterations must be positive and tolerance non-negative.")
        if self.gist_kmeans_init not in {"kmeans++", "sample"}:
            raise ValueError(f"Unsupported gist_kmeans_init: {self.gist_kmeans_init}")
        if self.gist_kmeans_empty_cluster_policy not in {"farthest", "error"}:
            raise ValueError(
                "Unsupported gist_kmeans_empty_cluster_policy: "
                f"{self.gist_kmeans_empty_cluster_policy}"
            )
        self.gist_som_steps = int(self.gist_som_steps)
        if self.gist_som_steps < 0:
            raise ValueError("gist_som_steps must be non-negative.")
        if min(
            float(self.gist_som_learning_rate),
            float(self.gist_som_final_learning_rate),
            float(self.gist_som_neighborhood_radius),
            float(self.gist_som_final_neighborhood_radius),
        ) < 0:
            raise ValueError("SOM rates and neighborhood radii must be non-negative.")
        if self.gist_som_distance not in {"cosine", "euclidean"}:
            raise ValueError(f"Unsupported gist_som_distance: {self.gist_som_distance}")
        if self.gist_som_init != "sample" or self.gist_som_topology != "line":
            raise ValueError("The current SOM implementation supports sample initialization and line topology.")
        if self.gist_prototype_method != "farthest":
            raise ValueError(f"Unsupported gist_prototype_method: {self.gist_prototype_method}")
        if self.gist_prototype_init not in {"mean_nearest", "sample"}:
            raise ValueError(f"Unsupported gist_prototype_init: {self.gist_prototype_init}")
        if self.gist_prototype_distance not in {"cosine", "euclidean"}:
            raise ValueError(
                f"Unsupported gist_prototype_distance: {self.gist_prototype_distance}"
            )
        if self.gist_hybrid_global_mode not in {"mean", "last", "ref_end", "gru"}:
            raise ValueError(
                f"Unsupported gist_hybrid_global_mode: {self.gist_hybrid_global_mode}"
            )
        if (
            self.reference_level_gist_mode == "hybrid"
            and self.gist_hybrid_global_mode == "ref_end"
        ):
            raise ValueError("Reference-level hybrid gists cannot use a ref_end global gist.")
        if self.gist_hybrid_local_mode not in {"kmeans", "som", "prototype"}:
            raise ValueError(f"Unsupported gist_hybrid_local_mode: {self.gist_hybrid_local_mode}")
        self.gist_hybrid_global_count = int(self.gist_hybrid_global_count)
        if self.gist_hybrid_global_count <= 0:
            raise ValueError("gist_hybrid_global_count must be positive.")
        if not 0.0 <= float(self.gist_hybrid_min_cosine_separation) <= 2.0:
            raise ValueError("gist_hybrid_min_cosine_separation must be between zero and two.")
        if self.chunking_mode not in {"none", "fixed", "markers", "semantic"}:
            raise ValueError(f"Unsupported chunking_mode: {self.chunking_mode}")
        self.fixed_chunk_tokens = int(self.fixed_chunk_tokens)
        self.fixed_chunk_overlap_tokens = int(self.fixed_chunk_overlap_tokens)
        self.chunk_overlap_fraction = float(self.chunk_overlap_fraction)
        if self.fixed_chunk_tokens <= 0:
            raise ValueError("fixed_chunk_tokens must be positive.")
        if self.fixed_chunk_overlap_tokens < 0:
            raise ValueError("fixed_chunk_overlap_tokens must be non-negative.")
        if not 0.0 <= self.chunk_overlap_fraction < 1.0:
            raise ValueError("chunk_overlap_fraction must satisfy 0 <= value < 1.")
        if self.fixed_chunk_overlap_tokens and self.chunk_overlap_fraction:
            raise ValueError(
                "Set only one of fixed_chunk_overlap_tokens and chunk_overlap_fraction."
            )
        if self.resolved_chunk_overlap_tokens >= self.fixed_chunk_tokens:
            raise ValueError(
                "Resolved fixed chunk overlap must be smaller than fixed_chunk_tokens."
            )
        if self.overlap_materialization not in {"deduplicate", "keep_duplicates"}:
            raise ValueError(
                "overlap_materialization must be 'deduplicate' or 'keep_duplicates'."
            )
        if self.reference_encoding_strategy not in {
            "independent",
            "block_slice",
            "native_slice",
        }:
            raise ValueError(
                f"Unsupported reference_encoding_strategy: {self.reference_encoding_strategy}"
            )
        self.encoding_block_references = int(self.encoding_block_references)
        if self.encoding_block_references <= 0:
            raise ValueError("encoding_block_references must be positive.")
        self.encoding_overlap_fraction = float(self.encoding_overlap_fraction)
        if not 0.0 <= self.encoding_overlap_fraction < 1.0:
            raise ValueError("encoding_overlap_fraction must satisfy 0 <= value < 1.")
        if self.reference_position_mode not in {"local", "global"}:
            raise ValueError("reference_position_mode must be 'local' or 'global'.")
        if self.prompt_position_mode not in {"local", "historical"}:
            raise ValueError("prompt_position_mode must be 'local' or 'historical'.")
        if self.encoding_context_mode not in {
            "independent",
            "overlap",
            "historical_window",
        }:
            raise ValueError(
                "encoding_context_mode must be independent, overlap, or historical_window."
            )
        self.encoding_chunking = (
            ChunkingConfig.from_value(self.encoding_chunking)
            if self.encoding_chunking is not None
            else None
        )
        self.routing_chunking = (
            ChunkingConfig.from_value(self.routing_chunking)
            if self.routing_chunking is not None
            else None
        )
        encoding_size = self.encoding_chunking_config.chunk_tokens
        if encoding_size is not None and encoding_size > self.effective_model_max_context_tokens:
            raise ValueError(
                "encoding_chunking.chunk_tokens cannot exceed model_max_context_tokens."
            )
        if self.reference_overflow_policy not in {"truncate", "error"}:
            raise ValueError(
                f"Unsupported reference_overflow_policy: {self.reference_overflow_policy}"
            )
        if self.kv_cache_residency not in {"gpu", "cpu"}:
            raise ValueError("kv_cache_residency must be 'gpu' or 'cpu'.")
        if self.kv_cache_residency == "cpu" and self.cache_build_mode != "detached":
            raise ValueError("CPU-resident K/V requires cache_build_mode='detached'.")
        self.marker_rules = tuple(str(value) for value in self.marker_rules)
        if self.chunking_mode == "semantic" and self.semantic_chunker is None:
            raise NotImplementedError(
                "chunking_mode='semantic' requires an explicit semantic_chunker implementation."
            )
        if self.detail_materialization not in {
            "selected_chunks",
            "full_reference",
            "gist_only",
            "native_gist_only",
            "logical_intervals",
            "gist_plus_logical_intervals",
        }:
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
        variants = {"custom", "td_sa", "td_pra", "tdx_pra", "td_layered_pra"}
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
        elif self.model_variant == "td_layered_pra":
            self.n_vanilla_layers = 0
            self.n_mixed_layers = 0
            self.pra_layer_ids = tuple(sorted({int(layer) for layer in self.pra_layer_ids}))
            if not self.pra_layer_ids:
                raise ValueError("td_layered_pra requires at least one pra_layer_id.")
            if self.pra_layer_ids[0] < 0 or self.pra_layer_ids[-1] >= self.n_layers:
                raise ValueError("pra_layer_ids must index configured decoder layers.")
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
class TrainConfig(CommonTrainConfig):
    """Generic training settings extended with PRA resolver/cache services."""

    experiment_name: str = "standalone_tiny"
    dataset_stage: str = "stage0_synthetic_memory"
    resolver_config: ResolverServiceConfig = field(default_factory=ResolverServiceConfig)
    cache_config: CacheServiceConfig = field(default_factory=CacheServiceConfig)

    def __post_init__(self) -> None:
        """Normalize nested resolver/cache settings accepted from YAML or Python."""
        super().__post_init__()
        self.resolver_config = ResolverServiceConfig.from_value(self.resolver_config)
        self.cache_config = CacheServiceConfig.from_value(self.cache_config)
