"""Native PyTorch AGILE3D-compatible multi-object click decoder.

This module adapts the interaction/query mechanisms from AGILE3D to frozen
LitePT dec0 tokens. The MinkowskiEngine backbone is intentionally absent:
LitePT already supplies token features, token coordinates, and the
point-to-token mapping required by the decoder.

The attention block layout, position/click encodings, mask-query maximum, and
loss names follow AGILE3D. See THIRD_PARTY_NOTICES.md for the upstream MIT
attribution.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _with_pos(tensor: Tensor, pos: Tensor | None) -> Tensor:
    return tensor if pos is None else tensor + pos


class _SelfAttentionLayer(nn.Module):
    """AGILE3D's pre/post-norm self-attention block in batch-first form."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        *,
        dropout: float = 0.0,
        pre_norm: bool = False,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = bool(pre_norm)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(
        self,
        target: Tensor,
        *,
        target_mask: Tensor | None = None,
        query_pos: Tensor | None = None,
    ) -> Tensor:
        if self.pre_norm:
            normalized = self.norm(target)
            q = _with_pos(normalized, query_pos)
            attended = self.attn(q, q, normalized, attn_mask=target_mask)[0]
            return target + self.dropout(attended)
        q = _with_pos(target, query_pos)
        attended = self.attn(q, q, target, attn_mask=target_mask)[0]
        return self.norm(target + self.dropout(attended))


class _CrossAttentionLayer(nn.Module):
    """AGILE3D's residual cross-attention block."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        *,
        dropout: float = 0.0,
        pre_norm: bool = False,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = bool(pre_norm)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(
        self,
        target: Tensor,
        memory: Tensor,
        *,
        memory_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
        query_pos: Tensor | None = None,
    ) -> Tensor:
        if self.pre_norm:
            normalized = self.norm(target)
            attended = self.attn(
                _with_pos(normalized, query_pos),
                _with_pos(memory, pos),
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )[0]
            return target + self.dropout(attended)
        attended = self.attn(
            _with_pos(target, query_pos),
            _with_pos(memory, pos),
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        return self.norm(target + self.dropout(attended))


class _FFNLayer(nn.Module):
    """AGILE3D feed-forward residual block."""

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        *,
        dropout: float = 0.0,
        pre_norm: bool = False,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = bool(pre_norm)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, target: Tensor) -> Tensor:
        if self.pre_norm:
            normalized = self.norm(target)
            update = self.linear2(F.relu(self.linear1(normalized)))
            return target + self.dropout(update)
        update = self.linear2(self.dropout(F.relu(self.linear1(target))))
        return self.norm(target + self.dropout(update))


class Agile3DClickDecoder(nn.Module):
    """Full AGILE3D-style multi-object decoder over one LitePT level."""

    schema = "delimit3d_agile3d_click_decoder/v1"

    def __init__(
        self,
        *,
        feature_dim: int = 72,
        hidden_dim: int = 128,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        num_decoders: int = 3,
        num_bg_queries: int = 10,
        dropout: float = 0.0,
        pre_norm: bool = False,
        max_click_events: int = 200,
        normalize_pos_enc: bool = True,
        gauss_scale: float = 1.0,
        aux: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % 2 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be even and divisible by num_heads")
        if max_click_events <= 0:
            raise ValueError("max_click_events must be positive")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.dim_feedforward = int(dim_feedforward)
        self.num_decoders = int(num_decoders)
        self.num_bg_queries = int(num_bg_queries)
        self.max_click_events = int(max_click_events)
        self.normalize_pos_enc = bool(normalize_pos_enc)
        self.aux = bool(aux)

        self.input_adapter = nn.Linear(self.feature_dim, self.hidden_dim)
        self.bg_query_feat = nn.Embedding(self.num_bg_queries, self.hidden_dim)
        self.bg_query_pos = nn.Embedding(self.num_bg_queries, self.hidden_dim)
        self.mask_embed_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.decoder_norm = nn.LayerNorm(self.hidden_dim)

        gauss = torch.empty(3, self.hidden_dim // 2)
        gauss.normal_()
        gauss.mul_(float(gauss_scale))
        self.register_buffer("gauss_B", gauss)
        self.register_buffer(
            "time_encode", self._make_time_encoding(self.max_click_events, self.hidden_dim)
        )

        self.c2s_attention = nn.ModuleList()
        self.c2c_attention = nn.ModuleList()
        self.ffn_attention = nn.ModuleList()
        self.s2c_attention = nn.ModuleList()
        for _ in range(self.num_decoders):
            self.c2s_attention.append(
                _CrossAttentionLayer(
                    self.hidden_dim,
                    self.num_heads,
                    dropout=dropout,
                    pre_norm=pre_norm,
                )
            )
            self.c2c_attention.append(
                _SelfAttentionLayer(
                    self.hidden_dim,
                    self.num_heads,
                    dropout=dropout,
                    pre_norm=pre_norm,
                )
            )
            self.ffn_attention.append(
                _FFNLayer(
                    self.hidden_dim,
                    self.dim_feedforward,
                    dropout=dropout,
                    pre_norm=pre_norm,
                )
            )
            self.s2c_attention.append(
                _CrossAttentionLayer(
                    self.hidden_dim,
                    self.num_heads,
                    dropout=dropout,
                    pre_norm=pre_norm,
                )
            )

    @staticmethod
    def _make_time_encoding(length: int, dimension: int) -> Tensor:
        positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, dimension, 2, dtype=torch.float32)
            * (-math.log(10000.0) / dimension)
        )
        encoding = torch.zeros(length, dimension, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * div)
        encoding[:, 1::2] = torch.cos(positions * div)
        return encoding

    @staticmethod
    def _normalize_clicks(
        clicks: Mapping[int | str, Sequence[int]],
        click_times: Mapping[int | str, Sequence[int]],
    ) -> tuple[dict[int, list[int]], dict[int, list[int]], int]:
        normalized: dict[int, list[int]] = {}
        normalized_times: dict[int, list[int]] = {}
        time_lookup = {int(key): values for key, values in click_times.items()}
        for key, values in clicks.items():
            object_id = int(key)
            ids = [int(value) for value in values]
            if object_id < 0:
                raise ValueError("click object IDs must be non-negative")
            normalized[object_id] = ids
            if object_id not in time_lookup:
                raise ValueError(f"missing click times for object {object_id}")
            times = [int(value) for value in time_lookup[object_id]]
            if len(times) != len(ids):
                raise ValueError(f"click/time length mismatch for object {object_id}")
            normalized_times[object_id] = times
        positive = sorted(key for key in normalized if key > 0)
        if not positive:
            raise ValueError("at least one foreground object is required")
        expected = list(range(1, len(positive) + 1))
        if positive != expected:
            raise ValueError(f"foreground object IDs must be contiguous 1..K, got {positive}")
        normalized.setdefault(0, [])
        normalized_times.setdefault(0, [])
        return normalized, normalized_times, len(positive)

    def _position_encoding(self, xyz: Tensor) -> Tensor:
        """Return normalized Fourier position encodings with shape [N, hidden]."""
        if xyz.ndim != 2 or xyz.shape[-1] != 3:
            raise ValueError(f"xyz must be [N,3], got {tuple(xyz.shape)}")
        with torch.no_grad():
            minimum = xyz.min(dim=0).values
            maximum = xyz.max(dim=0).values
            if self.normalize_pos_enc:
                scale = (maximum - minimum).clamp_min(1e-6)
                normalized = (xyz - minimum) / scale
            else:
                normalized = xyz
            projected = (normalized * (2.0 * math.pi)) @ self.gauss_B
            return torch.cat((projected.sin(), projected.cos()), dim=-1)

    def _query_inputs(
        self,
        scene_features: Tensor,
        scene_xyz: Tensor,
        clicks: Mapping[int | str, Sequence[int]],
        click_times: Mapping[int | str, Sequence[int]],
    ) -> tuple[Tensor, Tensor, list[int], int]:
        clicks_n, times_n, object_count = self._normalize_clicks(clicks, click_times)
        scene_count = int(scene_features.shape[0])
        fg_features: list[Tensor] = []
        fg_positions: list[Tensor] = []
        fg_splits: list[int] = []
        for object_id in range(1, object_count + 1):
            indices = clicks_n[object_id]
            if not indices:
                raise ValueError(f"foreground object {object_id} has no positive click")
            if min(indices) < 0 or max(indices) >= scene_count:
                raise IndexError(f"foreground click for object {object_id} is outside token range")
            index_tensor = torch.as_tensor(indices, device=scene_features.device, dtype=torch.long)
            time_tensor = torch.as_tensor(times_n[object_id], device=scene_features.device, dtype=torch.long)
            if bool((time_tensor < 0).any()) or bool((time_tensor >= self.max_click_events).any()):
                raise ValueError("click time exceeds temporal encoding capacity")
            fg_features.append(scene_features[index_tensor])
            fg_positions.append(
                self._position_encoding(scene_xyz[index_tensor])
                + self.time_encode[time_tensor].to(scene_features.device)
            )
            fg_splits.append(len(indices))

        bg_indices = clicks_n.get(0, [])
        bg_features = self.bg_query_feat.weight
        bg_positions = self.bg_query_pos.weight
        if bg_indices:
            if min(bg_indices) < 0 or max(bg_indices) >= scene_count:
                raise IndexError("background click is outside token range")
            index_tensor = torch.as_tensor(bg_indices, device=scene_features.device, dtype=torch.long)
            time_tensor = torch.as_tensor(times_n[0], device=scene_features.device, dtype=torch.long)
            if bool((time_tensor < 0).any()) or bool((time_tensor >= self.max_click_events).any()):
                raise ValueError("background click time exceeds temporal encoding capacity")
            bg_features = torch.cat((bg_features, scene_features[index_tensor]), dim=0)
            bg_positions = torch.cat(
                (
                    bg_positions,
                    self._position_encoding(scene_xyz[index_tensor])
                    + self.time_encode[time_tensor].to(scene_features.device),
                ),
                dim=0,
            )

        query_features = torch.cat(fg_features + [bg_features], dim=0)
        query_positions = torch.cat(fg_positions + [bg_positions], dim=0)
        if query_features.ndim != 2 or query_features.shape[-1] != self.hidden_dim:
            raise RuntimeError("query feature construction produced an invalid shape")
        return query_features, query_positions, fg_splits, int(bg_features.shape[0])

    def _mask_module(
        self,
        foreground_queries: Tensor,
        background_queries: Tensor,
        scene_features: Tensor,
        fg_splits: Sequence[int],
    ) -> tuple[Tensor, Tensor]:
        normalized_fg = self.decoder_norm(foreground_queries)
        fg_embeddings = self.mask_embed_head(normalized_fg)
        fg_products = scene_features @ fg_embeddings.transpose(0, 1)
        split_products = list(fg_products.split(list(fg_splits), dim=1))
        fg_masks = [product.max(dim=1, keepdim=True).values for product in split_products]
        fg_logits = torch.cat(fg_masks, dim=1)

        normalized_bg = self.decoder_norm(background_queries)
        bg_embeddings = self.mask_embed_head(normalized_bg)
        bg_logits = (scene_features @ bg_embeddings.transpose(0, 1)).max(dim=1, keepdim=True).values
        output = torch.cat((bg_logits, fg_logits), dim=1)

        labels = output.argmax(dim=1)
        query_count = foreground_queries.shape[0] + background_queries.shape[0]
        attention_mask = torch.ones(
            query_count,
            scene_features.shape[0],
            dtype=torch.bool,
            device=scene_features.device,
        )
        row = 0
        for object_id, split in enumerate(fg_splits, start=1):
            allowed = labels == object_id
            for _ in range(int(split)):
                mask_row = ~allowed
                if bool(mask_row.all()):
                    mask_row = torch.zeros_like(mask_row)
                attention_mask[row] = mask_row
                row += 1
        allowed_bg = labels == 0
        for _ in range(background_queries.shape[0]):
            mask_row = ~allowed_bg
            if bool(mask_row.all()):
                mask_row = torch.zeros_like(mask_row)
            attention_mask[row] = mask_row
            row += 1
        return output, attention_mask

    def forward(
        self,
        scene_tokens: Tensor,
        scene_xyz: Tensor,
        *,
        clicks: Mapping[int | str, Sequence[int]],
        click_times: Mapping[int | str, Sequence[int]],
    ) -> dict[str, Any]:
        """Predict one scene episode.

        scene_tokens and scene_xyz are [V,72] and [V,3]. Click indices refer
        to the token sequence. The returned logits are [V,K+1].
        """
        if scene_tokens.ndim != 2 or scene_tokens.shape[-1] != self.feature_dim:
            raise ValueError(f"scene_tokens must be [V,{self.feature_dim}]")
        if scene_xyz.shape != (scene_tokens.shape[0], 3):
            raise ValueError("scene_xyz must align with scene_tokens")
        scene = self.input_adapter(scene_tokens)
        query, query_pos, fg_splits, bg_count = self._query_inputs(
            scene, scene_xyz, clicks, click_times
        )
        foreground_count = sum(fg_splits)
        foreground = query[:foreground_count]
        background = query[foreground_count:]
        foreground_pos = query_pos[:foreground_count]
        background_pos = query_pos[foreground_count:]
        position = self._position_encoding(scene_xyz)

        predictions: list[Tensor] = []
        attention_mask: Tensor | None = None
        for stage in range(self.num_decoders):
            all_queries = torch.cat((foreground, background), dim=0)
            all_positions = torch.cat((foreground_pos, background_pos), dim=0)
            all_queries = self.c2s_attention[stage](
                all_queries.unsqueeze(1),
                scene.unsqueeze(1),
                memory_mask=attention_mask if attention_mask is not None else None,
                pos=position.unsqueeze(1),
                query_pos=all_positions.unsqueeze(1),
            ).squeeze(1)
            all_queries = self.c2c_attention[stage](
                all_queries.unsqueeze(1),
                query_pos=all_positions.unsqueeze(1),
            ).squeeze(1)
            all_queries = self.ffn_attention[stage](all_queries.unsqueeze(1)).squeeze(1)
            updated_scene = self.s2c_attention[stage](
                scene.unsqueeze(1),
                all_queries.unsqueeze(1),
                pos=all_positions.unsqueeze(1),
                query_pos=position.unsqueeze(1),
            ).squeeze(1)
            foreground, background = all_queries.split(
                [foreground_count, bg_count], dim=0
            )
            scene = updated_scene
            logits, attention_mask = self._mask_module(
                foreground, background, scene, fg_splits
            )
            predictions.append(logits)

        if not predictions:
            raise RuntimeError("decoder has no stages")
        output: dict[str, Any] = {
            "pred_masks": predictions[-1],
            "all_predictions": predictions,
        }
        if self.aux:
            output["aux_outputs"] = [{"pred_masks": value} for value in predictions[:-1]]
        return output


def click_loss_weights(
    scene_xyz: Tensor,
    clicks: Mapping[int | str, Sequence[int]],
    *,
    alpha: float = 0.8,
    beta: float = 2.0,
    radius: float = 0.3,
) -> Tensor:
    """AGILE3D click-local weights using all positive and negative clicks."""
    if radius <= 0 or alpha < 0 or beta < 0:
        raise ValueError("click weighting parameters must be non-negative")
    click_indices: list[int] = []
    for values in clicks.values():
        click_indices.extend(int(value) for value in values)
    if not click_indices:
        return torch.ones(scene_xyz.shape[0], device=scene_xyz.device, dtype=scene_xyz.dtype)
    index_tensor = torch.as_tensor(click_indices, device=scene_xyz.device, dtype=torch.long)
    if bool((index_tensor < 0).any()) or bool((index_tensor >= scene_xyz.shape[0]).any()):
        raise IndexError("click index outside scene token range")
    distances = torch.cdist(scene_xyz, scene_xyz[index_tensor]).min(dim=1).values
    clipped = distances.clamp(max=float(radius))
    return float(alpha) + (float(beta) - float(alpha)) * (1.0 - clipped / float(radius))


def multiclass_dice_loss(
    logits: Tensor,
    target: Tensor,
    *,
    weights: Tensor | None = None,
    eps: float = 1e-6,
) -> Tensor:
    """AGILE3D's multiclass soft-Dice loss with token-local weighting."""
    if logits.ndim != 2 or target.ndim != 1 or logits.shape[0] != target.shape[0]:
        raise ValueError("logits must be [V,C] and target must be [V]")
    probabilities = logits.softmax(dim=1)
    one_hot = F.one_hot(target.long(), num_classes=logits.shape[1]).to(probabilities.dtype)
    numerator = 2.0 * probabilities * one_hot
    denominator = probabilities + one_hot
    per_token = torch.where(
        numerator > eps,
        1.0 - (numerator + eps) / (denominator + eps),
        numerator * 0.0,
    ).mean(dim=1)
    if weights is not None:
        per_token = per_token * weights
    return per_token.mean()


def compute_agile3d_losses(
    outputs: Mapping[str, Any],
    target: Tensor,
    *,
    scene_xyz: Tensor,
    clicks: Mapping[int | str, Sequence[int]],
    bce_weight: float = 1.0,
    dice_weight: float = 2.0,
    alpha: float = 0.8,
    beta: float = 2.0,
    radius: float = 0.3,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return weighted loss and per-output CE/Dice diagnostics."""
    weights = click_loss_weights(
        scene_xyz, clicks, alpha=alpha, beta=beta, radius=radius
    )
    predictions = [outputs["pred_masks"]]
    predictions.extend(item["pred_masks"] for item in outputs.get("aux_outputs", []))
    details: dict[str, Tensor] = {}
    total = target.new_zeros((), dtype=torch.float32)
    for index, logits in enumerate(predictions):
        prefix = "main" if index == 0 else f"aux_{index - 1}"
        ce = (F.cross_entropy(logits, target.long(), reduction="none") * weights).mean()
        dice = multiclass_dice_loss(logits, target, weights=weights)
        details[f"{prefix}_loss_bce"] = ce
        details[f"{prefix}_loss_dice"] = dice
        total = total + float(bce_weight) * ce + float(dice_weight) * dice
    details["loss_bce"] = details["main_loss_bce"]
    details["loss_dice"] = details["main_loss_dice"]
    details["loss_total"] = total
    return total, details


def state_sha256(state: Mapping[str, Tensor]) -> str:
    """Hash a tensor state dict with names, dtypes, shapes, and raw bytes."""
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(str(key).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def save_initialization(
    path: str | Path,
    *,
    seed: int,
    decoder_kwargs: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the immutable shared decoder initialization."""
    path = Path(path).expanduser()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite decoder initialization: {path}")
    torch.manual_seed(int(seed))
    kwargs = dict(decoder_kwargs or {})
    decoder = Agile3DClickDecoder(**kwargs)
    state = {key: value.detach().cpu().clone() for key, value in decoder.state_dict().items()}
    tensor_hash = state_sha256(state)
    payload = {
        "schema": Agile3DClickDecoder.schema,
        "seed": int(seed),
        "decoder_kwargs": kwargs,
        "state_dict": state,
        "tensor_state_sha256": tensor_hash,
        "metadata": dict(metadata or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "path": str(path),
        "schema": payload["schema"],
        "seed": int(seed),
        "decoder_kwargs": kwargs,
        "tensor_state_sha256": tensor_hash,
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def load_initialization(
    path: str | Path,
    *,
    expected_kwargs: Mapping[str, Any] | None = None,
) -> tuple[Agile3DClickDecoder, dict[str, Any]]:
    path = Path(path).expanduser().resolve(strict=True)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema") != Agile3DClickDecoder.schema:
        raise ValueError(f"{path}: invalid AGILE3D decoder initialization")
    kwargs = dict(payload.get("decoder_kwargs") or {})
    for key, value in (expected_kwargs or {}).items():
        if kwargs.get(key) != value:
            raise ValueError(f"{path}: decoder setting {key} drifted")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError(f"{path}: missing decoder state")
    digest = state_sha256(state)
    if digest != payload.get("tensor_state_sha256"):
        raise ValueError(f"{path}: decoder initialization hash mismatch")
    decoder = Agile3DClickDecoder(**kwargs)
    decoder.load_state_dict(state, strict=True)
    return decoder, {
        "path": str(path),
        "seed": int(payload.get("seed", -1)),
        "decoder_kwargs": kwargs,
        "tensor_state_sha256": digest,
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "metadata": dict(payload.get("metadata") or {}),
    }


__all__ = [
    "Agile3DClickDecoder",
    "click_loss_weights",
    "compute_agile3d_losses",
    "load_initialization",
    "multiclass_dice_loss",
    "save_initialization",
    "state_sha256",
]

