"""Feature representation diagnostics used by debug observability."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _float(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        value = value.detach().float().reshape(-1)[0].item()
    return float(value)


def _subsample_indices(n: int, max_points: int | None, *, seed: int, device: torch.device) -> torch.Tensor:
    if max_points is None or n <= int(max_points):
        return torch.arange(n, device=device)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    idx = torch.randperm(n, generator=gen)[: int(max_points)]
    return idx.to(device=device)


def compute_feature_norm_stats(features: torch.Tensor) -> dict[str, float]:
    """Return norm mean/std for ``[N, C]`` feature tensors."""
    if features.ndim != 2:
        raise ValueError(f"features must be [N, C], got {tuple(features.shape)}")
    if features.numel() == 0:
        return {
            "point_feature_norm_mean": 0.0,
            "point_feature_norm_std": 0.0,
        }
    norms = torch.linalg.norm(features.detach().float(), dim=1)
    return {
        "point_feature_norm_mean": _float(norms.mean()),
        "point_feature_norm_std": _float(norms.std(unbiased=False)),
    }


def compute_pca_stats(
    features: torch.Tensor,
    *,
    max_points: int | None = 4096,
    seed: int = 0,
) -> dict[str, float]:
    """Compute top-3 PCA explained variance ratios on a subsample."""
    if features.ndim != 2:
        raise ValueError(f"features must be [N, C], got {tuple(features.shape)}")
    n, c = int(features.shape[0]), int(features.shape[1])
    if n < 3 or c < 3:
        return {
            "pca_explained_var_0": 0.0,
            "pca_explained_var_1": 0.0,
            "pca_explained_var_2": 0.0,
        }
    idx = _subsample_indices(n, max_points, seed=seed, device=features.device)
    x = features.detach().float()[idx].cpu()
    x = x - x.mean(dim=0, keepdim=True)
    _u, s, _v = torch.pca_lowrank(x, q=3, center=False, niter=3)
    total_var = x.var(dim=0, unbiased=False).sum().clamp_min(1e-12)
    explained = ((s ** 2) / max(int(x.shape[0]), 1) / total_var).tolist()
    explained = (explained + [0.0, 0.0, 0.0])[:3]
    return {
        "pca_explained_var_0": float(explained[0]),
        "pca_explained_var_1": float(explained[1]),
        "pca_explained_var_2": float(explained[2]),
    }


def compute_instance_feature_separation(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    min_points: int = 2,
    max_points_per_instance: int = 256,
    seed: int = 0,
) -> dict[str, float]:
    """Compare intra-instance and nearest inter-instance cosine distances."""
    if features.ndim != 2:
        raise ValueError(f"features must be [N, C], got {tuple(features.shape)}")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("labels must be [N] and match features")
    x = F.normalize(features.detach().float(), dim=1)
    labels = labels.detach().long().to(device=x.device)
    ids = [
        int(v.item())
        for v in labels.unique(sorted=True)
        if int(v.item()) > 0 and int((labels == int(v.item())).sum().item()) >= int(min_points)
    ]
    if not ids:
        return {
            "intra_instance_cosine_distance": 0.0,
            "nearest_inter_instance_cosine_distance": 0.0,
            "inter_over_intra_feature_separation": 0.0,
            "same_instance_cosine_mean": 0.0,
            "different_instance_cosine_mean": 0.0,
        }

    centroids: list[torch.Tensor] = []
    intra_vals: list[torch.Tensor] = []
    same_cos_vals: list[torch.Tensor] = []
    for offset, inst_id in enumerate(ids):
        mask = labels == inst_id
        idx = mask.nonzero(as_tuple=False).flatten()
        if idx.numel() > int(max_points_per_instance):
            gen = torch.Generator(device="cpu").manual_seed(int(seed) + offset)
            choice = torch.randperm(idx.numel(), generator=gen)[: int(max_points_per_instance)].to(idx.device)
            idx = idx[choice]
        feat_i = x[idx]
        centroid = F.normalize(feat_i.mean(dim=0, keepdim=True), dim=1).squeeze(0)
        centroids.append(centroid)
        cos = (feat_i * centroid).sum(dim=1)
        intra_vals.append(1.0 - cos)
        same_cos_vals.append(cos)

    intra = torch.cat(intra_vals) if intra_vals else x.new_zeros((0,))
    same_cos = torch.cat(same_cos_vals) if same_cos_vals else x.new_zeros((0,))
    inter = x.new_zeros((0,))
    different_cos = x.new_zeros((0,))
    if len(centroids) >= 2:
        c = torch.stack(centroids, dim=0)
        sim = c @ c.T
        sim.fill_diagonal_(-1.0)
        nearest_sim = sim.max(dim=1).values
        inter = 1.0 - nearest_sim
        different_cos = sim[sim > -0.999]
    intra_mean = _float(intra.mean()) if intra.numel() else 0.0
    inter_mean = _float(inter.mean()) if inter.numel() else 0.0
    return {
        "intra_instance_cosine_distance": intra_mean,
        "nearest_inter_instance_cosine_distance": inter_mean,
        "inter_over_intra_feature_separation": inter_mean / max(intra_mean, 1e-12),
        "same_instance_cosine_mean": _float(same_cos.mean()) if same_cos.numel() else 0.0,
        "different_instance_cosine_mean": _float(different_cos.mean()) if different_cos.numel() else 0.0,
    }


def compute_feature_diagnostics(
    features: torch.Tensor,
    labels: torch.Tensor | None = None,
    *,
    max_pca_points: int | None = 4096,
    seed: int = 0,
) -> dict[str, float]:
    """Combined feature diagnostics for lightweight step logging."""
    stats = compute_feature_norm_stats(features)
    stats.update(compute_pca_stats(features, max_points=max_pca_points, seed=seed))
    if labels is not None:
        stats.update(compute_instance_feature_separation(features, labels, seed=seed))
    return stats


def pca_rgb(
    features: torch.Tensor,
    *,
    max_fit_points: int | None = 200_000,
    seed: int = 0,
    q_low: float = 0.01,
    q_high: float = 0.99,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return robust PCA RGB colors in ``[0, 1]`` for all feature rows."""
    if features.ndim != 2:
        raise ValueError(f"features must be [N, C], got {tuple(features.shape)}")
    n, c = int(features.shape[0]), int(features.shape[1])
    if n < 3 or c < 3:
        rgb = torch.zeros((n, 3), dtype=torch.float32)
        return rgb, {"explained_variance_ratio": [0.0, 0.0, 0.0]}
    idx = _subsample_indices(n, max_fit_points, seed=seed, device=features.device)
    x = features.detach().float().cpu()
    x_fit = x[idx.cpu()]
    mean = x_fit.mean(dim=0, keepdim=True)
    centered = x_fit - mean
    _u, s, v = torch.pca_lowrank(centered, q=3, center=False, niter=5)
    projected = (x - mean) @ v[:, :3]
    lo = torch.quantile(projected, float(q_low), dim=0, keepdim=True)
    hi = torch.quantile(projected, float(q_high), dim=0, keepdim=True)
    rgb = ((projected - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)
    total_var = centered.var(dim=0, unbiased=False).sum().clamp_min(1e-12)
    explained = ((s ** 2) / max(int(centered.shape[0]), 1) / total_var).tolist()
    return rgb, {"explained_variance_ratio": (explained + [0.0, 0.0, 0.0])[:3]}
