"""Shared helpers for point-feature PCA visualization.

PCA colors are a projection aid only: component signs/colors are arbitrary,
and instance information can be hidden outside the top variance directions.
Use the feature diagnostics alongside rendered panels.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def stabilize_pca_sign(components: torch.Tensor) -> torch.Tensor:
    """Flip component signs deterministically for repeatable colors."""
    v = components.clone()
    for col in range(v.shape[1]):
        pivot = torch.argmax(v[:, col].abs())
        if v[pivot, col] < 0:
            v[:, col] *= -1
    return v


def _randperm(n: int, *, device: torch.device, seed: int) -> torch.Tensor:
    try:
        generator = torch.Generator(device=device).manual_seed(int(seed))
        return torch.randperm(n, generator=generator, device=device)
    except (RuntimeError, TypeError):
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        return torch.randperm(n, generator=generator, device="cpu").to(device)


def fit_pca(
    features: torch.Tensor,
    *,
    fit_mask: torch.Tensor | None = None,
    q: int = 3,
    niter: int = 5,
    max_fit_points: int | None = 200_000,
    seed: int = 0,
) -> dict[str, torch.Tensor | list[float] | int | bool]:
    """Fit a centered PCA basis and keep the mean for consistent transforms."""
    x = features.detach().float()
    if x.ndim != 2:
        raise ValueError(f"Need a 2D feature tensor, got {tuple(x.shape)}")

    if fit_mask is not None:
        mask = fit_mask.detach().to(device=x.device).bool()
        if mask.shape[0] != x.shape[0]:
            raise ValueError(f"fit_mask length {mask.shape[0]} != features length {x.shape[0]}")
        x_fit = x[mask] if bool(mask.any()) else x
    else:
        x_fit = x

    if max_fit_points is not None and x_fit.shape[0] > int(max_fit_points):
        idx = _randperm(x_fit.shape[0], device=x_fit.device, seed=seed)[: int(max_fit_points)]
        x_fit = x_fit[idx]

    q_eff = min(int(q), x_fit.shape[0], x_fit.shape[1])
    if q_eff <= 0:
        raise ValueError(f"Cannot PCA feature tensor with shape {tuple(features.shape)}")

    mean = x_fit.mean(dim=0, keepdim=True)
    centered = x_fit - mean
    _u, s, v = torch.pca_lowrank(centered, q=q_eff, center=False, niter=int(niter))
    v = stabilize_pca_sign(v)
    total_var = centered.var(dim=0, unbiased=False).sum().clamp_min(1e-12)
    explained = ((s.detach().cpu() ** 2) / max(centered.shape[0], 1) / total_var.detach().cpu()).tolist()
    return {
        "mean": mean.detach(),
        "components": v.detach(),
        "explained_variance_ratio": explained,
        "q": q_eff,
        "centered": True,
    }


def project_pca(
    features: torch.Tensor,
    pca: dict[str, torch.Tensor | list[float] | int | bool],
    *,
    num_components: int | None = None,
) -> torch.Tensor:
    x = features.detach().float()
    mean = pca["mean"]
    components = pca["components"]
    if not isinstance(mean, torch.Tensor) or not isinstance(components, torch.Tensor):
        raise TypeError("pca must contain tensor mean and components")
    mean = mean.to(device=x.device, dtype=x.dtype)
    components = components.to(device=x.device, dtype=x.dtype)
    if num_components is not None:
        components = components[:, : int(num_components)]
    return (x - mean) @ components


def pca_rgb_from_fit(
    features: torch.Tensor,
    pca: dict[str, torch.Tensor | list[float] | int | bool],
    *,
    q_low: float = 0.01,
    q_high: float = 0.99,
    robust_scale: bool = True,
    global_scale: bool = False,
    brightness: float = 1.0,
) -> torch.Tensor:
    rgb = project_pca(features, pca, num_components=3)
    if rgb.shape[1] < 3:
        rgb = F.pad(rgb, (0, 3 - rgb.shape[1]))

    if global_scale:
        lo = rgb.min()
        hi = rgb.max()
    elif robust_scale:
        lo = torch.quantile(rgb, float(q_low), dim=0, keepdim=True)
        hi = torch.quantile(rgb, float(q_high), dim=0, keepdim=True)
    else:
        lo = rgb.min(dim=0, keepdim=True).values
        hi = rgb.max(dim=0, keepdim=True).values

    rgb = (rgb - lo) / (hi - lo).clamp_min(1e-6)
    return (rgb * float(brightness)).clamp(0.0, 1.0)


def pca_rgb(
    features: torch.Tensor,
    *,
    fit_mask: torch.Tensor | None = None,
    q: int = 3,
    niter: int = 5,
    q_low: float = 0.01,
    q_high: float = 0.99,
    robust_scale: bool = True,
    global_scale: bool = False,
    brightness: float = 1.0,
    max_fit_points: int | None = 200_000,
    seed: int = 0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    pca = fit_pca(
        features,
        fit_mask=fit_mask,
        q=q,
        niter=niter,
        max_fit_points=max_fit_points,
        seed=seed,
    )
    rgb = pca_rgb_from_fit(
        features,
        pca,
        q_low=q_low,
        q_high=q_high,
        robust_scale=robust_scale,
        global_scale=global_scale,
        brightness=brightness,
    )
    return rgb, {
        "explained_variance_ratio": pca["explained_variance_ratio"],
        "q": pca["q"],
        "centered": True,
        "q_low": q_low,
        "q_high": q_high,
        "robust_scale": robust_scale,
        "global_scale": global_scale,
        "brightness": brightness,
        "max_fit_points": max_fit_points,
    }


def feature_instance_diagnostics(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    min_points: int = 10,
    nn_sample_points: int = 4096,
    nn_k: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
    """Measure instance separation in the feature space, not in PCA RGB space."""
    x = F.normalize(features.detach().float().cpu(), dim=1)
    labels_cpu = labels.detach().cpu().long()
    ids, counts = labels_cpu.unique(sorted=True, return_counts=True)
    instance_ids = [
        int(inst_id.item())
        for inst_id, count in zip(ids, counts)
        if int(inst_id.item()) >= 0 and int(count.item()) >= int(min_points)
    ]
    if not instance_ids:
        return {
            "num_instances": 0,
            "mean_within_instance_cosine_similarity": None,
            "mean_between_instance_centroid_cosine_similarity": None,
            "mean_nearest_inter_instance_centroid_cosine_similarity": None,
            "max_nearest_inter_instance_centroid_cosine_similarity": None,
            "separation_gap_mean": None,
            "separation_gap_nearest": None,
            "knn_label_purity_top1": None,
            "knn_label_purity_topk": None,
        }

    centroids: list[torch.Tensor] = []
    within: list[float] = []
    keep_mask = torch.zeros_like(labels_cpu, dtype=torch.bool)
    for inst_id in instance_ids:
        mask = labels_cpu == inst_id
        keep_mask |= mask
        feat_i = x[mask]
        centroid = F.normalize(feat_i.mean(dim=0, keepdim=True), dim=1).squeeze(0)
        centroids.append(centroid)
        within.extend(((feat_i * centroid).sum(dim=1)).tolist())

    within_mean = float(np.mean(within)) if within else None
    between_mean: float | None = None
    nearest_mean: float | None = None
    nearest_max: float | None = None
    if len(centroids) >= 2:
        c = torch.stack(centroids, dim=0)
        sim = c @ c.T
        off_diag = ~torch.eye(sim.shape[0], dtype=torch.bool)
        between_mean = float(sim[off_diag].mean().item())
        sim_for_nearest = sim.clone()
        sim_for_nearest.fill_diagonal_(-float("inf"))
        nearest = sim_for_nearest.max(dim=1).values
        nearest_mean = float(nearest.mean().item())
        nearest_max = float(nearest.max().item())

    purity_top1: float | None = None
    purity_topk: float | None = None
    valid_idx = torch.nonzero(keep_mask, as_tuple=False).flatten()
    if valid_idx.numel() > max(1, nn_k):
        if valid_idx.numel() > int(nn_sample_points):
            rng = np.random.default_rng(int(seed))
            choice = np.sort(rng.choice(valid_idx.numel(), size=int(nn_sample_points), replace=False))
            valid_idx = valid_idx[torch.from_numpy(choice).long()]
        xs = x[valid_idx]
        ys = labels_cpu[valid_idx]
        k_eff = min(int(nn_k), xs.shape[0] - 1)
        sim = xs @ xs.T
        sim.fill_diagonal_(-float("inf"))
        neigh = torch.topk(sim, k=k_eff, dim=1).indices
        same = ys[:, None] == ys[neigh]
        purity_top1 = float(same[:, 0].float().mean().item())
        purity_topk = float(same.float().mean().item())

    return {
        "num_instances": len(instance_ids),
        "min_points": int(min_points),
        "mean_within_instance_cosine_similarity": within_mean,
        "mean_between_instance_centroid_cosine_similarity": between_mean,
        "mean_nearest_inter_instance_centroid_cosine_similarity": nearest_mean,
        "max_nearest_inter_instance_centroid_cosine_similarity": nearest_max,
        "separation_gap_mean": None if within_mean is None or between_mean is None else within_mean - between_mean,
        "separation_gap_nearest": None if within_mean is None or nearest_mean is None else within_mean - nearest_mean,
        "knn_label_purity_top1": purity_top1,
        "knn_label_purity_topk": purity_topk,
        "knn_k": int(nn_k),
        "knn_sample_points": int(min(valid_idx.numel(), int(nn_sample_points))),
    }
