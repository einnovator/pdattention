"""Hybrid global-plus-local gist strategy composed from existing implementations."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import ComputedGists
from .common import empty_gists, mean_cosine_separation, validate_points
from .kmeans import KMeansGistStrategy
from .prototype import PrototypeGistStrategy
from .single import SingleGistStrategy
from .som import SOMGistStrategy


class HybridGistStrategy:
    """Combine one global single gist with specialized local prototypes."""

    _LOCAL = {
        "kmeans": KMeansGistStrategy,
        "som": SOMGistStrategy,
        "prototype": PrototypeGistStrategy,
    }

    @staticmethod
    def _deduplicate(k, v, minimum_separation: float):
        if k.shape[0] < 2:
            return k, v
        kept = []
        for index in range(k.shape[0]):
            if not kept:
                kept.append(index)
                continue
            similarities = F.normalize(k[index : index + 1], dim=-1) @ F.normalize(
                k[torch.tensor(kept, device=k.device)], dim=-1
            ).transpose(0, 1)
            if bool(((1.0 - similarities) > minimum_separation + 1e-7).all()):
                kept.append(index)
        indices = torch.tensor(kept, device=k.device)
        return k[indices], v[indices] if v is not None else None

    def compute(self, *, keys, values, num_gists, config, context) -> ComputedGists:
        validate_points(keys, values)
        requested = int(num_gists)
        if requested <= 0 or keys.shape[0] == 0:
            return empty_gists(keys, values, mode="hybrid", requested_gists=requested, actual_gists=0)
        target_count = min(requested, int(keys.shape[0]))
        global_result = SingleGistStrategy(config.gist_hybrid_global_mode).compute(
            keys=keys,
            values=values,
            num_gists=min(int(config.gist_hybrid_global_count), target_count),
            config=config,
            context=context,
        )
        remaining = max(target_count - int(global_result.k.shape[0]), 0)
        local_class = self._LOCAL.get(config.gist_hybrid_local_mode)
        if local_class is None:
            raise ValueError(f"Unsupported hybrid local mode: {config.gist_hybrid_local_mode}")
        local_result = local_class().compute(
            keys=keys,
            values=values,
            num_gists=remaining,
            config=config,
            context=context,
        )
        gist_k = torch.cat((global_result.k, local_result.k), dim=0)
        gist_v = (
            torch.cat((global_result.v, local_result.v), dim=0)
            if global_result.v is not None and local_result.v is not None
            else global_result.v if local_result.k.shape[0] == 0 else None
        )
        if config.gist_hybrid_deduplicate:
            gist_k, gist_v = self._deduplicate(
                gist_k,
                gist_v,
                float(config.gist_hybrid_min_cosine_separation),
            )
        return ComputedGists(
            k=gist_k,
            v=gist_v,
            metadata={
                "mode": "hybrid",
                "requested_gists": requested,
                "actual_gists": int(gist_k.shape[0]),
                "global_mode": config.gist_hybrid_global_mode,
                "local_mode": config.gist_hybrid_local_mode,
                "global_metadata": global_result.metadata,
                "local_metadata": local_result.metadata,
                "mean_cosine_separation": mean_cosine_separation(gist_k),
            },
        )
