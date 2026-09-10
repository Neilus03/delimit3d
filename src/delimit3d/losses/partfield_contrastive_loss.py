"""PartField's symmetric relative multi-negative contrastive objective."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from delimit3d.data.partfield_contrastive import ContrastiveTripletBatch


class PartFieldContrastiveCriterion(nn.Module):
    """Train point embeddings from proposal-local positive/negative relations.

    The objective is symmetric in the two positive endpoints. For each endpoint,
    its paired positive occupies logit 0 and all proposal-complement points are
    negatives. Proposal IDs never become classifier targets and therefore may
    be inconsistent across frames, scenes, sources, and granularities.
    """

    def __init__(
        self,
        *,
        temperature: float = 0.07,
        learnable_temperature: bool = True,
        min_temperature: float = 0.01,
        max_temperature: float = 1.0,
        eps: float = 1e-8,
        symmetric_feature_hard_mining: bool = False,
    ) -> None:
        super().__init__()
        if not 0 < float(min_temperature) <= float(temperature) <= float(max_temperature):
            raise ValueError(
                "Expected 0 < min_temperature <= temperature <= max_temperature"
            )
        self.min_temperature = float(min_temperature)
        self.max_temperature = float(max_temperature)
        self.eps = float(eps)
        self.symmetric_feature_hard_mining = bool(
            symmetric_feature_hard_mining
        )
        initial = torch.tensor(math.log(float(temperature)), dtype=torch.float32)
        if learnable_temperature:
            self.log_temperature = nn.Parameter(initial)
        else:
            self.register_buffer("log_temperature", initial)

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(
            min=self.min_temperature,
            max=self.max_temperature,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        batch: ContrastiveTripletBatch,
    ) -> dict[str, Any]:
        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must have shape [N, D], got {tuple(embeddings.shape)}"
            )
        if batch.positive_pairs.ndim != 2 or batch.positive_pairs.shape[1] != 2:
            raise ValueError("positive_pairs must have shape [P, 2]")
        if batch.negative_indices.ndim != 2:
            raise ValueError("negative_indices must have shape [P, M]")
        if batch.positive_pairs.shape[0] != batch.negative_indices.shape[0]:
            raise ValueError("positive pair and negative rows must match")
        if int(batch.num_feature_hard_negatives) < 0:
            raise ValueError("num_feature_hard_negatives must be non-negative")
        feature_candidates = batch.feature_candidate_indices
        if int(batch.num_feature_hard_negatives) > 0:
            if feature_candidates is None:
                raise ValueError(
                    "feature_candidate_indices are required for feature-hard mining"
                )
            if feature_candidates.ndim != 2:
                raise ValueError("feature_candidate_indices must have shape [P, Q]")
            if feature_candidates.shape[0] != batch.positive_pairs.shape[0]:
                raise ValueError("feature candidate and positive pair rows must match")
            if feature_candidates.shape[1] < int(batch.num_feature_hard_negatives):
                raise ValueError("Feature candidate pool is smaller than hard-negative count")
        if batch.num_pairs == 0 or batch.num_negatives == 0:
            raise ValueError("Contrastive batch must contain pairs and negatives")

        index_chunks = [batch.positive_pairs.reshape(-1)]
        if batch.negative_indices.numel() > 0:
            index_chunks.append(batch.negative_indices.reshape(-1))
        if feature_candidates is not None:
            index_chunks.append(feature_candidates.reshape(-1))
        all_indices = torch.cat(index_chunks)
        if int(all_indices.min()) < 0 or int(all_indices.max()) >= embeddings.shape[0]:
            raise IndexError("Contrastive point index is outside the embedding tensor")

        z = F.normalize(embeddings, dim=-1, eps=self.eps)
        a = z[batch.positive_pairs[:, 0]]
        b = z[batch.positive_pairs[:, 1]]
        negative_parts_a: list[torch.Tensor] = []
        negative_parts_b: list[torch.Tensor] = []
        if batch.negative_indices.shape[1] > 0:
            base_negative_features = z[batch.negative_indices]
            negative_parts_a.append(base_negative_features)
            negative_parts_b.append(base_negative_features)
        if int(batch.num_feature_hard_negatives) > 0:
            assert feature_candidates is not None
            with torch.no_grad():
                candidate_features = z.detach()[feature_candidates]
                candidate_similarity_a = torch.einsum(
                    "pd,pqd->pq", a.detach(), candidate_features
                )
                hard_local_a = torch.topk(
                    candidate_similarity_a,
                    k=int(batch.num_feature_hard_negatives),
                    dim=1,
                    largest=True,
                ).indices
                hard_indices_a = torch.gather(
                    feature_candidates, 1, hard_local_a
                )
                if self.symmetric_feature_hard_mining:
                    candidate_similarity_b = torch.einsum(
                        "pd,pqd->pq", b.detach(), candidate_features
                    )
                    hard_local_b = torch.topk(
                        candidate_similarity_b,
                        k=int(batch.num_feature_hard_negatives),
                        dim=1,
                        largest=True,
                    ).indices
                    hard_indices_b = torch.gather(
                        feature_candidates, 1, hard_local_b
                    )
                else:
                    hard_indices_b = hard_indices_a
            negative_parts_a.append(z[hard_indices_a])
            negative_parts_b.append(z[hard_indices_b])
        negatives_a = torch.cat(negative_parts_a, dim=1)
        negatives_b = torch.cat(negative_parts_b, dim=1)

        positive_cosine = (a * b).sum(dim=-1)
        negative_cosine_a = torch.einsum("pd,pmd->pm", a, negatives_a)
        negative_cosine_b = torch.einsum("pd,pmd->pm", b, negatives_b)

        temperature = self.temperature.to(device=embeddings.device, dtype=embeddings.dtype)
        logits_a = torch.cat([positive_cosine[:, None], negative_cosine_a], dim=1)
        logits_b = torch.cat([positive_cosine[:, None], negative_cosine_b], dim=1)
        logits_a = logits_a / temperature
        logits_b = logits_b / temperature
        target = torch.zeros(batch.num_pairs, dtype=torch.long, device=embeddings.device)
        loss_a = F.cross_entropy(logits_a, target)
        loss_b = F.cross_entropy(logits_b, target)
        loss = 0.5 * (loss_a + loss_b)

        with torch.no_grad():
            negative_cosine = 0.5 * (
                negative_cosine_a.mean() + negative_cosine_b.mean()
            )
            mean_positive = positive_cosine.mean()
            ranking = 0.5 * (
                (positive_cosine[:, None] > negative_cosine_a).float().mean()
                + (positive_cosine[:, None] > negative_cosine_b).float().mean()
            )
            hardest_negative = torch.maximum(
                negative_cosine_a.max(dim=1).values,
                negative_cosine_b.max(dim=1).values,
            )
            hardest_ranking = (positive_cosine > hardest_negative).float().mean()

        return {
            "loss_total": loss,
            "loss_contrastive": float(loss.detach()),
            "temperature": float(temperature.detach()),
            "positive_cosine_mean": float(mean_positive),
            "negative_cosine_mean": float(negative_cosine),
            "cosine_gap": float(mean_positive - negative_cosine),
            "triplet_ranking_accuracy": float(ranking),
            "hardest_negative_ranking_accuracy": float(hardest_ranking),
            "num_positive_pairs": batch.num_pairs,
            "num_negatives_per_pair": batch.num_negatives,
            "num_base_negatives_per_pair": batch.num_base_negatives,
            "num_feature_hard_negatives_per_pair": int(
                batch.num_feature_hard_negatives
            ),
        }
