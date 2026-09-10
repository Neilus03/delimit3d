"""Configurable runtime identity and external artifact roots.

The Python import package remains ``delimit3d`` so installed code is stable,
while run names, report labels, and default artifact locations are controlled by
``DELIMIT3D_NAME``.  This lets a fork rename the project without editing every
launcher or report generator.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

DEFAULT_NAME = "Delimit3D"
NAME_ENV = "DELIMIT3D_NAME"
WORK_ROOT_ENV = "DELIMIT3D_WORK_ROOT"
ARTIFACT_ROOT_ENV = "DELIMIT3D_ARTIFACT_ROOT"


def project_name() -> str:
    value = os.environ.get(NAME_ENV) or os.environ.get("DELIMIT3D_PROJECT_NAME")
    value = (value or DEFAULT_NAME).strip()
    return value or DEFAULT_NAME


def project_slug() -> str:
    value = project_name().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value or "delimit3d"


def work_root() -> Path:
    return Path(os.path.expandvars(os.path.expanduser(
        os.environ.get(WORK_ROOT_ENV, "/cluster/work/igp_psr/nedela")
    ))).resolve()


def artifact_root() -> Path:
    configured = os.environ.get(ARTIFACT_ROOT_ENV)
    if configured:
        return Path(os.path.expandvars(os.path.expanduser(configured))).resolve()
    return work_root() / project_slug()


def data_root(name: str) -> Path:
    env_name = f"DELIMIT3D_{str(name).upper()}_ROOT"
    configured = os.environ.get(env_name)
    if configured:
        return Path(os.path.expandvars(os.path.expanduser(configured))).resolve()
    defaults = {
        "structured3d": work_root() / "structured3d_raw",
        "scannet": work_root() / "scannet_raw",
        "scannetpp": work_root() / "scannetpp_data",
        "litept": work_root() / "LitePT",
        "unsam": work_root() / "UnSAMv2",
    }
    try:
        return defaults[str(name).lower()].resolve()
    except KeyError as exc:
        raise ValueError(f"Unknown data root {name!r}") from exc


def identity_record() -> dict[str, str]:
    return {
        "project_name": project_name(),
        "project_slug": project_slug(),
        "name_env": NAME_ENV,
        "artifact_root": str(artifact_root()),
    }


__all__ = [
    "DEFAULT_NAME",
    "NAME_ENV",
    "artifact_root",
    "data_root",
    "identity_record",
    "project_name",
    "project_slug",
    "work_root",
]
