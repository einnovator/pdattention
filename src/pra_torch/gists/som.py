"""Small deterministic self-organizing-map strategy for local gist construction."""

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


class SOMGistStrategy:
    """Fit a line-topology SOM locally, then pair V gists through winner assignments."""

    def compute(self, *, keys, values, num_gists, config, context) -> ComputedGists:
        del context
        validate_points(keys, values)
        requested = int(num_gists)
        count = min(max(requested, 0), int(keys.shape[0]))
        if count == 0:
            return empty_gists(keys, values, mode="som", requested_gists=requested, actual_gists=0)
        if config.gist_som_topology != "line":
            raise ValueError(f"Unsupported SOM topology: {config.gist_som_topology}")
        work = normalize_rows(keys) if config.gist_som_normalize else keys
        generator = local_generator(config.gist_som_seed)
        if config.gist_som_init != "sample":
            raise ValueError(f"Unsupported SOM initialization: {config.gist_som_init}")
        initial = torch.randperm(work.shape[0], generator=generator)[:count].to(work.device)
        prototypes = work[initial].clone()
        steps = int(config.gist_som_steps)
        positions = torch.arange(count, device=work.device, dtype=work.dtype)
        for step in range(steps):
            point_index = int(torch.randint(work.shape[0], (1,), generator=generator).item())
            point = work[point_index]
            winner = int(
                assign_to_centers(
                    point.unsqueeze(0),
                    prototypes,
                    distance=config.gist_som_distance,
                )[0].item()
            )
            progress = step / max(steps - 1, 1)
            learning_rate = (
                float(config.gist_som_learning_rate) * (1.0 - progress)
                + float(config.gist_som_final_learning_rate) * progress
            )
            radius = (
                float(config.gist_som_neighborhood_radius) * (1.0 - progress)
                + float(config.gist_som_final_neighborhood_radius) * progress
            )
            if radius > 0:
                influence = torch.exp(-((positions - winner) ** 2) / (2.0 * radius**2))
            else:
                influence = (positions == winner).to(work.dtype)
            prototypes = prototypes + learning_rate * influence[:, None] * (point - prototypes)
            if config.gist_som_normalize:
                prototypes = normalize_rows(prototypes)
        assignments = assign_to_centers(
            work,
            prototypes,
            distance=config.gist_som_distance,
        )
        _, gist_v, occupied, occupancy = paired_means(keys, values, assignments, count)
        gist_k = prototypes[torch.tensor(occupied, device=prototypes.device)]
        if config.gist_som_normalize and gist_k.shape[0]:
            gist_k = normalize_rows(gist_k)
        return ComputedGists(
            k=gist_k,
            v=gist_v,
            metadata={
                "mode": "som",
                "requested_gists": requested,
                "actual_gists": int(gist_k.shape[0]),
                "occupancy": occupancy,
                "steps": steps,
                "mean_cosine_separation": mean_cosine_separation(gist_k),
            },
        )
