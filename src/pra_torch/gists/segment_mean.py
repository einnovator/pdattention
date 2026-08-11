"""Deterministic contiguous sub-chunk means for parameter-free multi-gist routing."""

from __future__ import annotations

import torch

from .base import ComputedGists
from .common import empty_gists, mean_cosine_separation, validate_points


class SegmentMeanGistStrategy:
    """Mean-pool balanced contiguous token segments into ``[G,D]`` gists."""

    def compute(self, *, keys, values, num_gists, config, context) -> ComputedGists:
        """Preserve source order while producing at most one gist per source token."""
        del config, context
        validate_points(keys, values)
        requested = int(num_gists)
        if keys.shape[0] == 0:
            return empty_gists(
                keys,
                values,
                mode="segment_mean",
                requested_gists=requested,
                actual_gists=0,
                occupancy=[],
                segment_token_spans=[],
            )

        count = min(requested, int(keys.shape[0]))
        base, remainder = divmod(int(keys.shape[0]), count)
        spans: list[tuple[int, int]] = []
        cursor = 0
        for index in range(count):
            end = cursor + base + int(index < remainder)
            spans.append((cursor, end))
            cursor = end

        key_gists = torch.stack([keys[start:end].mean(dim=0) for start, end in spans])
        value_gists = (
            torch.stack([values[start:end].mean(dim=0) for start, end in spans])
            if values is not None
            else None
        )
        return ComputedGists(
            k=key_gists,
            v=value_gists,
            metadata={
                "mode": "segment_mean",
                "requested_gists": requested,
                "actual_gists": count,
                "occupancy": [end - start for start, end in spans],
                "segment_token_spans": [list(span) for span in spans],
                "mean_cosine_separation": mean_cosine_separation(key_gists),
            },
        )
