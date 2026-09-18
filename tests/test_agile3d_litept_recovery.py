"""Recovery must retain the last checkpoint if storage fails during a write."""
import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/training/run_agile3d_litept_sstar_rgb3.py"
    spec = importlib.util.spec_from_file_location("agile3d_recovery_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_interrupted_save_preserves_previous_checkpoint(tmp_path, monkeypatch, runner):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000])
    path = tmp_path / "checkpoint.pth"
    config = {"experiment_id": "test", "_config_sha256": "test"}
    runner.save_checkpoint(path, model, optimizer, scheduler, 11, 2880, config)
    before = path.read_bytes()
    saved = torch.load(path, weights_only=False)
    assert saved["epoch"] == 11 and saved["step"] == 2880
    assert saved["rng_state"]

    def interrupted_save(payload, handle):
        handle.write(b"incomplete checkpoint")
        raise OSError("simulated storage failure")

    monkeypatch.setattr(torch, "save", interrupted_save)
    with pytest.raises(OSError, match="storage failure"):
        runner.save_checkpoint(path, model, optimizer, scheduler, 12, 3120, config)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_staging_preserves_config_hash_and_training_settings(tmp_path, monkeypatch, runner):
    config = Path(__file__).resolve().parents[1] / "configs/training/agile3d_litept_sstar_rgb3_scannet40.yaml"
    monkeypatch.delenv("AGILE3D_STAGED_ROOT", raising=False)
    original = runner.load_config(config)
    monkeypatch.setenv("AGILE3D_STAGED_ROOT", str(tmp_path))
    staged = runner.load_config(config)
    assert staged["_config_sha256"] == original["_config_sha256"]
    for key in original.keys() - {"paths"}:
        assert staged[key] == original[key]
    assert staged["paths"]["output_root"] == original["paths"]["output_root"]
    assert staged["paths"]["scan_folder"] == str(tmp_path / "ScanNet/scans")
