"""Strategy registry for single- and multi-gist PRA routing representations."""

from __future__ import annotations

from .base import ComputedGists, GistContext, GistStrategy
from .common import GistScore, projected_tokens, score_gist_set
from .hybrid import HybridGistStrategy
from .kmeans import KMeansGistStrategy
from .prototype import PrototypeGistStrategy
from .segment_mean import SegmentMeanGistStrategy
from .single import GRUGistPooler, SingleGistStrategy
from .som import SOMGistStrategy


def compute_gists(*, keys, values, mode, num_gists, config, context) -> ComputedGists:
    """Dispatch one gist-construction call without mixing strategy implementations."""
    if mode in {"mean", "last", "ref_end", "gru"}:
        strategy = SingleGistStrategy(mode)
    elif mode == "segment_mean":
        strategy = SegmentMeanGistStrategy()
    elif mode == "kmeans":
        strategy = KMeansGistStrategy()
    elif mode == "som":
        strategy = SOMGistStrategy()
    elif mode == "prototype":
        strategy = PrototypeGistStrategy()
    elif mode == "hybrid":
        strategy = HybridGistStrategy()
    else:
        raise ValueError(f"Unsupported gist mode: {mode}")
    return strategy.compute(
        keys=keys,
        values=values,
        num_gists=num_gists,
        config=config,
        context=context,
    )


__all__ = [
    "ComputedGists",
    "GRUGistPooler",
    "GistContext",
    "GistScore",
    "GistStrategy",
    "SegmentMeanGistStrategy",
    "compute_gists",
    "projected_tokens",
    "score_gist_set",
]
