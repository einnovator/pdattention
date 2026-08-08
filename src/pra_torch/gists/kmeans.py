"""Deterministic local k-means routing gists with shared K/V assignments."""

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


class KMeansGistStrategy:
    """Cluster in key space and aggregate values from the same token assignments."""

    @staticmethod
    def _initialize(points: torch.Tensor, count: int, config) -> torch.Tensor:
        generator = local_generator(config.gist_kmeans_seed)
        if config.gist_kmeans_init == "sample":
            indices = torch.randperm(points.shape[0], generator=generator)[:count]
            return points[indices.to(points.device)].clone()
        if config.gist_kmeans_init != "kmeans++":
            raise ValueError(f"Unsupported k-means initialization: {config.gist_kmeans_init}")
        first = int(torch.randint(points.shape[0], (1,), generator=generator).item())
        chosen = [first]
        while len(chosen) < count:
            centers = points[torch.tensor(chosen, device=points.device)]
            distances = torch.cdist(points, centers).pow(2).min(dim=1).values
            distances[torch.tensor(chosen, device=points.device)] = 0
            total = float(distances.sum().detach().cpu())
            if total <= 0:
                remaining = [index for index in range(points.shape[0]) if index not in chosen]
                chosen.append(remaining[0])
            else:
                next_index = int(
                    torch.multinomial(
                        distances.detach().cpu() / total,
                        1,
                        generator=generator,
                    ).item()
                )
                if next_index in chosen:
                    next_index = next(index for index in range(points.shape[0]) if index not in chosen)
                chosen.append(next_index)
        return points[torch.tensor(chosen, device=points.device)].clone()

    def compute(self, *, keys, values, num_gists, config, context) -> ComputedGists:
        del context
        validate_points(keys, values)
        requested = int(num_gists)
        count = min(max(requested, 0), int(keys.shape[0]))
        if count == 0:
            return empty_gists(keys, values, mode="kmeans", requested_gists=requested, actual_gists=0)
        work = normalize_rows(keys) if config.gist_kmeans_normalize else keys
        centers = self._initialize(work, count, config)
        iterations = 0
        for iterations in range(1, int(config.gist_kmeans_max_iters) + 1):
            assignments = assign_to_centers(work, centers, distance="euclidean")
            updated = []
            for index in range(count):
                members = work[assignments == index]
                if members.shape[0]:
                    updated.append(members.mean(dim=0))
                    continue
                if config.gist_kmeans_empty_cluster_policy == "error":
                    raise ValueError(f"Empty k-means cluster {index}.")
                nearest = torch.cdist(work, centers).min(dim=1).values
                updated.append(work[int(nearest.argmax().item())])
            next_centers = torch.stack(updated)
            shift = float((next_centers - centers).norm(dim=1).max().detach().cpu())
            centers = next_centers
            if shift <= float(config.gist_kmeans_tol):
                break
        assignments = assign_to_centers(work, centers, distance="euclidean")
        gist_k, gist_v, _, occupancy = paired_means(keys, values, assignments, count)
        if config.gist_kmeans_normalize and gist_k.shape[0]:
            gist_k = normalize_rows(gist_k)
        return ComputedGists(
            k=gist_k,
            v=gist_v,
            metadata={
                "mode": "kmeans",
                "requested_gists": requested,
                "actual_gists": int(gist_k.shape[0]),
                "occupancy": occupancy,
                "iterations": iterations,
                "mean_cosine_separation": mean_cosine_separation(gist_k),
            },
        )
