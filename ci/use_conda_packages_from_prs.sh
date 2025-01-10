# Copyright (c) 2025, NVIDIA CORPORATION.
# TODO(jameslamb): remove this file when https://github.com/rapidsai/raft/pull/2531 is merged

RAFT_COMMIT="76d788bfb688335e9d0fab3ee83afc71a1599a7c"

RAFT_CPP_CHANNEL=$(rapids-get-pr-conda-artifact raft 2531 cpp "${RAFT_COMMIT:0:7}")
RAFT_PYTHON_CHANNEL=$(rapids-get-pr-conda-artifact raft 2531 python "${RAFT_COMMIT:0:7}")

conda config --system --add channels "${RAFT_CPP_CHANNEL}"
conda config --system --add channels "${RAFT_PYTHON_CHANNEL}"
