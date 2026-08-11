"""Public PRA-HF API for bounded sparse native-K/V memory."""

from .config import PRAConfig
from .evaluation import evaluate_router_features
from .model import GenerationResult, PRAForCausalLM, ReferenceHandle
from .router import PRARouter

__version__ = "0.2.0rc1"

__all__ = [
    "GenerationResult",
    "PRAConfig",
    "PRAForCausalLM",
    "PRARouter",
    "ReferenceHandle",
    "evaluate_router_features",
]
