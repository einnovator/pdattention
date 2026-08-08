"""Diversity-based prototype routing gists with paired K/V regions."""

from __future__ import annotations

import torch

from .base import ComputedGists
from .common import (
    assign_to_centers,
    empty_gists,
    local_generator,
    mean_cosine_separation,
    normalize_rows,
    paired_means,
    validate_points,
)


class PrototypeGistStrategy:
    """Select deterministic farthest-point representatives in normalized key space."""

    def compute(self, *, keys, values, num_gists, config, context) -> ComputedGists:
        del context
        validate_points(keys, values)
        requested = int(num_gists)
        count = min(max(requested, 0), int(keys.shape[0]))
        if count == 0:
            return empty_gists(keys, values, mode="prototype", requested_gists=requested, actual_gists=0)
        work = normalize_rows(keys) if config.gist_prototype_normalize else keys
        if config.gist_prototype_method != "farthest":
            raise ValueError(f"Unsupported prototype method: {config.gist_prototype_method}")
        if config.gist_prototype_init == "mean_nearest":
            center = work.mean(dim=0, keepdim=True)
            first = int(torch.cdist(work, center).argmin().item())
        elif config.gist_prototype_init == "sample":
            generator = local_generator(config.gist_prototype_seed)
            first = int(torch.randint(work.shape[0], (1,), generator=generator).item())
        else:
            raise ValueError(f"Unsupported prototype initialization: {config.gist_prototype_init}")
        selected = [first]
        while len(selected) < count:
            centers = work[torch.tensor(selected, device=work.device)]
            distances = torch.cdist(work, centers).min(dim=1).values
            distances[torch.tensor(selected, device=work.device)] = -1
            selected.append(int(distances.argmax().item()))
        centers = work[torch.tensor(selected, device=work.device)]
        assignments = assign_to_centers(
            work,
            centers,
            distance=config.gist_prototype_distance,
        )
        mean_k, gist_v, occupied, occupancy = paired_means(keys, values, assignments, count)
        gist_k = mean_k if config.gist_prototype_refine else keys[torch.tensor(selected, device=keys.device)][occupied]
        if config.gist_prototype_normalize and gist_k.shape[0]:
            gist_k = normalize_rows(gist_k)
        return ComputedGists(
            k=gist_k,
            v=gist_v,
            metadata={
                "mode": "prototype",
                "requested_gists": requested,
                "actual_gists": int(gist_k.shape[0]),
                "occupancy": occupancy,
                "selected_point_indices": [selected[index] for index in occupied],
                "mean_cosine_separation": mean_cosine_separation(gist_k),
            },
        )
