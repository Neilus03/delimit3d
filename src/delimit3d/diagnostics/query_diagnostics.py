"""Pure query, score, mask, and matching diagnostics.

The functions here are intentionally model-agnostic and operate on small
tensor inputs so they can be unit-tested without LitePT or a training pack.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _finite_float(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        value = value.detach().float().reshape(-1)[0].item()
    return float(value)


def _percentiles(values: torch.Tensor, qs: tuple[float, ...]) -> list[float]:
    x = values.detach().float().flatten()
    if x.numel() == 0:
        return [0.0 for _ in qs]
    return [
        _finite_float(torch.quantile(x, float(q)))
        for q in qs
    ]


def _as_long_tensor(
    value: torch.Tensor | list[int] | tuple[int, ...] | Any | None,
    *,
    device: torch.device,
) -> torch.Tensor:
    if value is None:
        return torch.empty(0, dtype=torch.long, device=device)
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.long).flatten()
    return torch.as_tensor(value, dtype=torch.long, device=device).flatten()


def _as_float_tensor(
    value: torch.Tensor | list[float] | tuple[float, ...] | Any | None,
    *,
    device: torch.device,
) -> torch.Tensor:
    if value is None:
        return torch.empty(0, dtype=torch.float32, device=device)
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32).flatten()
    return torch.as_tensor(value, dtype=torch.float32, device=device).flatten()


def binary_mask_iou_matrix(masks: torch.Tensor) -> torch.Tensor:
    """Return pairwise IoU for boolean masks shaped ``[K, N]``."""
    if masks.ndim != 2:
        raise ValueError(f"masks must be [K, N], got {tuple(masks.shape)}")
    k = int(masks.shape[0])
    if k == 0:
        return masks.new_zeros((0, 0), dtype=torch.float32)
    m = masks.bool().float()
    inter = m @ m.T
    area = m.sum(dim=1)
    union = area[:, None] + area[None, :] - inter
    return inter / union.clamp_min(1.0)


def compute_duplicate_iou_stats(
    mask_logits: torch.Tensor,
    score_logits: torch.Tensor | None = None,
    *,
    mask_threshold: float = 0.5,
    topk: int = 50,
) -> dict[str, float]:
    """Compute duplicate overlap among top-scoring predicted masks."""
    if mask_logits.ndim != 2:
        raise ValueError(f"mask_logits must be [Q, N], got {tuple(mask_logits.shape)}")
    q = int(mask_logits.shape[0])
    if q <= 1:
        return {
            "duplicate_iou_topk_mean": 0.0,
            "duplicate_iou_topk_max": 0.0,
        }
    k = min(max(int(topk), 1), q)
    if score_logits is None:
        idx = torch.arange(q, device=mask_logits.device)[:k]
    else:
        idx = score_logits.detach().float().topk(k).indices
    masks = mask_logits.detach()[idx].sigmoid() >= float(mask_threshold)
    iou = binary_mask_iou_matrix(masks)
    tri = torch.triu(torch.ones_like(iou, dtype=torch.bool), diagonal=1)
    vals = iou[tri]
    if vals.numel() == 0:
        return {
            "duplicate_iou_topk_mean": 0.0,
            "duplicate_iou_topk_max": 0.0,
        }
    return {
        "duplicate_iou_topk_mean": _finite_float(vals.mean()),
        "duplicate_iou_topk_max": _finite_float(vals.max()),
    }


def compute_query_embedding_stats(
    query_embed: torch.Tensor | None,
    score_logits: torch.Tensor | None = None,
    *,
    topk: int = 50,
) -> dict[str, float]:
    """Return mean pairwise cosine similarity for all/top-k query embeddings."""
    if query_embed is None:
        return {}
    if query_embed.ndim != 2:
        raise ValueError(f"query_embed must be [Q, D], got {tuple(query_embed.shape)}")
    q = int(query_embed.shape[0])
    if q <= 1:
        return {
            "query_embed_cosine_mean": 0.0,
            "query_embed_cosine_topk_mean": 0.0,
        }
    emb = F.normalize(query_embed.detach().float(), dim=1)
    sim = emb @ emb.T
    tri = torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)
    out = {"query_embed_cosine_mean": _finite_float(sim[tri].mean())}
    k = min(max(int(topk), 1), q)
    if score_logits is None:
        idx = torch.arange(q, device=emb.device)[:k]
    else:
        idx = score_logits.detach().float().topk(k).indices
    if k <= 1:
        out["query_embed_cosine_topk_mean"] = 0.0
    else:
        top_sim = emb[idx] @ emb[idx].T
        top_tri = torch.triu(torch.ones_like(top_sim, dtype=torch.bool), diagonal=1)
        out["query_embed_cosine_topk_mean"] = _finite_float(top_sim[top_tri].mean())
    return out


def compute_query_score_mask_stats(
    mask_logits: torch.Tensor,
    score_logits: torch.Tensor,
    query_embed: torch.Tensor | None = None,
    *,
    mask_threshold: float = 0.5,
    min_points_per_proposal: int = 30,
    topk: int = 50,
) -> dict[str, float]:
    """Summarize score distribution, mask areas, dead queries, and duplicates."""
    if mask_logits.ndim != 2:
        raise ValueError(f"mask_logits must be [Q, N], got {tuple(mask_logits.shape)}")
    if score_logits.ndim != 1:
        raise ValueError(f"score_logits must be [Q], got {tuple(score_logits.shape)}")
    if int(mask_logits.shape[0]) != int(score_logits.shape[0]):
        raise ValueError("mask_logits and score_logits query counts differ")

    q, n = int(mask_logits.shape[0]), int(mask_logits.shape[1])
    scores = score_logits.detach().float().sigmoid()
    logits = score_logits.detach().float()
    masks = mask_logits.detach().float().sigmoid() >= float(mask_threshold)
    areas = masks.sum(dim=1).float()
    k = min(max(int(topk), 1), max(q, 1))
    top_idx = scores.topk(k).indices if q else torch.empty(0, dtype=torch.long, device=scores.device)
    top_areas = areas[top_idx] if top_idx.numel() else areas.new_zeros((0,))
    dead = (scores < 0.01) & (areas < int(min_points_per_proposal))
    huge = areas > (0.5 * float(max(n, 1)))

    score_p10, score_p50, score_p90, score_p99 = _percentiles(scores, (0.10, 0.50, 0.90, 0.99))
    logit_p10, logit_p50, logit_p90 = _percentiles(logits, (0.10, 0.50, 0.90))
    area_p10, area_p50, area_p90 = _percentiles(areas, (0.10, 0.50, 0.90))
    top_area_p10, top_area_p50, top_area_p90 = _percentiles(top_areas, (0.10, 0.50, 0.90))

    stats = {
        "num_queries": float(q),
        "dead_query_fraction": _finite_float(dead.float().mean()) if q else 0.0,
        "topk_score_prob_mean": _finite_float(scores[top_idx].mean()) if top_idx.numel() else 0.0,
        "prob_mean": _finite_float(scores.mean()) if q else 0.0,
        "prob_std": _finite_float(scores.std(unbiased=False)) if q else 0.0,
        "prob_p10": score_p10,
        "prob_p50": score_p50,
        "prob_p90": score_p90,
        "prob_p99": score_p99,
        "logit_mean": _finite_float(logits.mean()) if q else 0.0,
        "logit_std": _finite_float(logits.std(unbiased=False)) if q else 0.0,
        "logit_p10": logit_p10,
        "logit_p50": logit_p50,
        "logit_p90": logit_p90,
        "area_p10": area_p10,
        "area_p50": area_p50,
        "area_p90": area_p90,
        "area_min": _finite_float(areas.min()) if q else 0.0,
        "area_max": _finite_float(areas.max()) if q else 0.0,
        "topk_area_p10": top_area_p10,
        "topk_area_p50": top_area_p50,
        "topk_area_p90": top_area_p90,
        "empty_mask_fraction": _finite_float((areas <= 0).float().mean()) if q else 0.0,
        "huge_mask_fraction": _finite_float(huge.float().mean()) if q else 0.0,
    }
    stats.update(compute_duplicate_iou_stats(
        mask_logits,
        score_logits,
        mask_threshold=mask_threshold,
        topk=topk,
    ))
    stats.update(compute_query_embedding_stats(query_embed, score_logits, topk=topk))
    return stats


def compute_matching_calibration_stats(
    score_logits: torch.Tensor,
    *,
    matched_pred_indices: torch.Tensor | list[int] | tuple[int, ...] | Any | None = None,
    matched_target_indices: torch.Tensor | list[int] | tuple[int, ...] | Any | None = None,
    matched_ious: torch.Tensor | list[float] | tuple[float, ...] | Any | None = None,
    high_score_unmatched_threshold: float = 0.5,
) -> dict[str, float]:
    """Summarize matched IoUs and score separation between matched/unmatched queries."""
    if score_logits.ndim != 1:
        raise ValueError(f"score_logits must be [Q], got {tuple(score_logits.shape)}")
    device = score_logits.device
    q = int(score_logits.shape[0])
    scores = score_logits.detach().float().sigmoid()
    pred_idx = _as_long_tensor(matched_pred_indices, device=device)
    pred_idx = pred_idx[(pred_idx >= 0) & (pred_idx < q)]
    ious = _as_float_tensor(matched_ious, device=device)
    if ious.numel() != pred_idx.numel():
        ious = ious[: pred_idx.numel()]
    matched_mask = torch.zeros(q, dtype=torch.bool, device=device)
    if pred_idx.numel():
        matched_mask[pred_idx] = True
    unmatched_mask = ~matched_mask
    matched_scores = scores[matched_mask]
    unmatched_scores = scores[unmatched_mask]
    high_unmatched = unmatched_scores >= float(high_score_unmatched_threshold)

    iou_p25, iou_p50, iou_p75 = _percentiles(ious, (0.25, 0.50, 0.75))
    matched_mean = _finite_float(matched_scores.mean()) if matched_scores.numel() else 0.0
    unmatched_mean = _finite_float(unmatched_scores.mean()) if unmatched_scores.numel() else 0.0
    return {
        "matched_queries": float(pred_idx.numel()),
        "matched_query_fraction": float(pred_idx.numel() / max(q, 1)),
        "matched_iou_p25": iou_p25,
        "matched_iou_p50": iou_p50,
        "matched_iou_p75": iou_p75,
        "matched_iou_mean": _finite_float(ious.mean()) if ious.numel() else 0.0,
        "unmatched_queries": float(max(q - int(pred_idx.numel()), 0)),
        "score_prob_mean_matched": matched_mean,
        "score_prob_mean_unmatched": unmatched_mean,
        "score_gap_matched_minus_unmatched": matched_mean - unmatched_mean,
        "high_score_unmatched_count": float(high_unmatched.sum().item()),
        "high_score_unmatched_fraction": (
            _finite_float(high_unmatched.float().mean()) if unmatched_scores.numel() else 0.0
        ),
    }
