from __future__ import annotations

import torch

from delimit3d.evaluation.agile3d_decoder import (
    Agile3DClickDecoder,
    compute_agile3d_losses,
    save_initialization,
    state_sha256,
)


def _decoder() -> Agile3DClickDecoder:
    return Agile3DClickDecoder(
        feature_dim=8,
        hidden_dim=16,
        num_heads=4,
        dim_feedforward=32,
        num_decoders=3,
        num_bg_queries=2,
        max_click_events=20,
        aux=True,
    )


def _clicks() -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    return {"0": [], "1": [1, 2], "2": [6]}, {"0": [], "1": [0, 1], "2": [0]}


def test_multi_object_foreground_background_shapes() -> None:
    torch.manual_seed(4)
    decoder = _decoder()
    features = torch.randn(12, 8)
    xyz = torch.randn(12, 3)
    clicks, times = _clicks()
    output = decoder(features, xyz, clicks=clicks, click_times=times)
    assert output["pred_masks"].shape == (12, 3)
    assert len(output["aux_outputs"]) == 2
    assert all(value["pred_masks"].shape == (12, 3) for value in output["aux_outputs"])


def test_negative_background_click_is_consumed() -> None:
    decoder = _decoder()
    features = torch.randn(10, 8)
    xyz = torch.randn(10, 3)
    clicks = {"0": [8], "1": [1], "2": [5]}
    times = {"0": [2], "1": [0], "2": [0]}
    output = decoder(features, xyz, clicks=clicks, click_times=times)
    assert output["pred_masks"].shape == (10, 3)
    assert torch.isfinite(output["pred_masks"]).all()


def test_loss_and_backward_are_finite() -> None:
    torch.manual_seed(5)
    decoder = _decoder()
    features = torch.randn(14, 8)
    xyz = torch.randn(14, 3)
    clicks, times = _clicks()
    target = torch.zeros(14, dtype=torch.long)
    target[1:4] = 1
    target[6:8] = 2
    output = decoder(features, xyz, clicks=clicks, click_times=times)
    loss, details = compute_agile3d_losses(
        output, target, scene_xyz=xyz, clicks=clicks
    )
    assert torch.isfinite(loss)
    assert set(("loss_bce", "loss_dice", "loss_total")) <= set(details)
    loss.backward()
    assert any(parameter.grad is not None for parameter in decoder.parameters())


def test_shared_initialization_hash_is_reproducible(tmp_path) -> None:
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    kwargs = {
        "feature_dim": 8,
        "hidden_dim": 16,
        "num_heads": 4,
        "dim_feedforward": 32,
        "num_decoders": 3,
        "num_bg_queries": 2,
        "max_click_events": 20,
    }
    r1 = save_initialization(first, seed=20260911, decoder_kwargs=kwargs)
    r2 = save_initialization(second, seed=20260911, decoder_kwargs=kwargs)
    assert r1["tensor_state_sha256"] == r2["tensor_state_sha256"]
    assert state_sha256(torch.load(first, weights_only=False)["state_dict"]) == r1["tensor_state_sha256"]

