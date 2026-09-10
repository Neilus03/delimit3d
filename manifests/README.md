# Manifests

Commit small, machine-readable records that identify an experiment. A valid
GPU run records:

- `project_name` and `project_slug` from `DELIMIT3D_NAME`;
- the exact Git commit and resolved configuration;
- hashes for every input/source manifest and checkpoint;
- the external artifact root and Slurm job IDs;
- the host/GPU type and environment;
- the declared question, controlled change, expected compute, and success gate.

Raw arrays, checkpoints, logs, figures, and scene bundles remain outside Git.
Use a new run directory for every experiment; never append unrelated runs to an
existing output tree.
