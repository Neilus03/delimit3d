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


def test_joint_evaluator_requires_dynamic_joint_checkpoint_and_official_panels() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/evaluation/scannet40_agile3d_joint_eval_v1.yaml").read_text()
    )
    source = (ROOT / "scripts/evaluation/run_scannet40_joint_eval.py").read_text()
    assert config["evaluation"]["panels"] == ["MO", "SO"]
    assert config["evaluation"]["checkpoint_sha256"]
    assert config["evaluation"]["training_manifest_sha256"]
    assert "encoder_state_dict" in source
    assert "dynamic validation feature cache" in source
    assert "official_lines" in source
    assert "raw_object_ious" in source
