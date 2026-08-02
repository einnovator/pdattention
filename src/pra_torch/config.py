from dataclasses import dataclass, field


@dataclass
class PRAConfig:
    """Model and PRA architecture hyperparameters."""

    vocab_size: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    n_vanilla_layers: int = 0
    n_mixed_layers: int = 0
    max_seq_len: int = 128
    dropout: float = 0.0
    model_variant: str = "custom"

    # PRA-specific
    pra_layer_ids: tuple[int, ...] = (2, 3)
    top_k_refs: int = 2
    trigger_threshold: float = 0.2
    use_cross_attention_memory: bool = True
    use_concat_memory: bool = False
    memory_alpha: float = 0.5

    # Training
    batch_size: int = 8
    lr: float = 3e-4
    steps: int = 500
    device: str = "cuda"

    def __post_init__(self) -> None:
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
