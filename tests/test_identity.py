from __future__ import annotations

from delimit3d.identity import artifact_root, identity_record, project_name, project_slug


def test_runtime_name_controls_display_slug_and_default_artifact_root(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DELIMIT3D_NAME", "My Delimit Experiment")
    monkeypatch.setenv("DELIMIT3D_WORK_ROOT", str(tmp_path))
    monkeypatch.delenv("DELIMIT3D_ARTIFACT_ROOT", raising=False)

    assert project_name() == "My Delimit Experiment"
    assert project_slug() == "my-delimit-experiment"
    assert artifact_root() == tmp_path / "my-delimit-experiment"
    assert identity_record() == {
        "project_name": "My Delimit Experiment",
        "project_slug": "my-delimit-experiment",
        "name_env": "DELIMIT3D_NAME",
        "artifact_root": str(tmp_path / "my-delimit-experiment"),
    }


def test_explicit_artifact_root_wins_without_changing_runtime_name(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DELIMIT3D_NAME", "Delimit3D")
    monkeypatch.setenv("DELIMIT3D_ARTIFACT_ROOT", str(tmp_path / "fixed"))

    assert project_slug() == "delimit3d"
    assert artifact_root() == tmp_path / "fixed"
