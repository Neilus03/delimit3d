# Reproducibility protocol

Every run must have a unique external artifact directory and a committed
manifest with the resolved configuration, code commit, source/checkpoint
hashes, runtime, host/GPU type, and Slurm job IDs. Generated files are never
committed to the source repository.

The public and adapted arms share the same RGBN6 input construction, scene
coordinate normalization, representative-first 2 cm voxelization, query
selection, candidate-point definition, and metric implementation. Ground truth
selects the prompt and computes metrics; it is never passed into the frozen
LitePT forward.

The primary frozen readout reports ranked AP, fixed-threshold IoU, precision,
recall, F1, oracle-threshold IoU, and scene-paired bootstrap intervals. The
learned-decoder protocol adds one-click and simulated correction-click curves.
Results from these levels must never be merged into one headline number.
