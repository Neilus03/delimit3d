from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_joint_config_is_single_delimit3d_all_learnable_arm() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/evaluation/scannet40_agile3d_joint_v1.yaml").read_text()
    )
    assert config["experiment_id"] == "delimit3d_scannet40_agile3d_joint_v1"
    assert config["encoder"]["trainable"] is True
    assert config["encoder_adaptation_uses_scannet_labels"] is True
    assert config["optimizer"]["gpu_count"] == 1
    assert config["optimizer"]["encoder_lr"] == config["optimizer"]["decoder_lr"] == 1.0e-4
    assert config["precision"]["initial_scale"] == 1.0
    assert config["precision"]["growth_interval"] == 1000000


def test_joint_runner_has_separate_equal_lr_groups_and_full_checkpoints() -> None:
    source = (ROOT / "scripts/evaluation/run_scannet40_joint.py").read_text()
    assert '"name": "encoder"' in source
    assert '"name": "decoder"' in source
    assert '"encoder_frozen": False' in source
    assert "update % 250 == 0" in source
    assert "encoder_grad_norm" in source
    assert "_float32_loss_outputs" in source
    assert "_gradient_diagnostics" in source
