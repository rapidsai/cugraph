#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# initialize PIP_CONSTRAINT
source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX=$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")

# download wheels, store the directories holding them in variables

# cudf
CUDF_COMMIT=23d0852b79e2b53feb9f948b2e3e892d7cfb0707
CUDF_WHEELHOUSE=$(
  rapids-get-pr-artifact cudf 21671 python wheel --pkg_name cudf --stable "${CUDF_COMMIT}"
)
DASK_CUDF_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="dask_cudf_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-get-pr-artifact cudf 21671 python wheel "${CUDF_COMMIT}"
)
LIBCUDF_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-artifact cudf 21671 cpp wheel "${CUDF_COMMIT}"
)
PYLIBCUDF_WHEELHOUSE=$(
  rapids-get-pr-artifact cudf 21671 python wheel --pkg_name pylibcudf --stable "${CUDF_COMMIT}"
)

# cuvs
# (not needed because it's pulled w/ CPM and we aren't actually changing its source code for this)

# kvikio
KVIKIO_COMMIT=5c9e2564a701b87aa01a4ac99863b00fb5483078
LIBKVIKIO_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="libkvikio_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-artifact kvikio 942 cpp wheel "${KVIKIO_COMMIT}"
)
KVIKIO_WHEELHOUSE=$(
  rapids-get-pr-artifact kvikio 942 python wheel --pkg_name kvikio --stable "${KVIKIO_COMMIT}"
)

# raft
RAFT_COMMIT=c285d96d34b6f86415bd1502d6f2d232317bed98
LIBRAFT_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="libraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-artifact raft 2971 cpp wheel "${RAFT_COMMIT}"
)
PYLIBRAFT_WHEELHOUSE=$(
  rapids-get-pr-artifact raft 2971 python wheel --pkg_name pylibraft --stable "${RAFT_COMMIT}"
)
RAFT_DASK_WHEELHOUSE=$(
  rapids-get-pr-artifact raft 2971 python wheel --pkg_name raft_dask --stable "${RAFT_COMMIT}"
)

# rmm
RMM_COMMIT=e36df7200d5c05302c8fefbcb94a89aa634ef2b7
LIBRMM_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="librmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-artifact rmm 2270 cpp wheel "${RMM_COMMIT}"
)
RMM_WHEELHOUSE=$(
  rapids-get-pr-artifact rmm 2270 python wheel --pkg_name rmm --stable "${RMM_COMMIT}"
)

# ucxx
UCXX_COMMIT=347de3a7c1c694fe8509ee7a6aa3d11f5e674476
DISTRIBUTED_UCXX_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-get-pr-artifact --pkg_name distributed-ucxx ucxx 604 python wheel
)
LIBUCXX_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="libucxx_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-artifact ucxx 604 cpp wheel "${UCXX_COMMIT}"
)
UCXX_WHEELHOUSE=$(
  rapids-get-pr-artifact ucxx 604 python wheel --pkg_name ucxx --stable "${UCXX_COMMIT}"
)

# write a pip constraints file saying e.g. "whenever you encounter a requirement for 'librmm-cu12', use this wheel"
cat >> "${PIP_CONSTRAINT}" <<EOF
cudf-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${CUDF_WHEELHOUSE}"/cudf*.whl)
dask-cudf-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${DASK_CUDF_WHEELHOUSE}"/dask_cudf*.whl)
libcudf-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf*.whl)
pylibcudf-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${PYLIBCUDF_WHEELHOUSE}"/pylibcudf*.whl)

kvikio-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${KVIKIO_WHEELHOUSE}"/kvikio_*.whl)
libkvikio-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBKVIKIO_WHEELHOUSE}"/libkvikio*.whl)

libraft-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBRAFT_WHEELHOUSE}"/libraft*.whl)
pylibraft-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${PYLIBRAFT_WHEELHOUSE}"/pylibraft*.whl)
raft-dask-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${RAFT_DASK_WHEELHOUSE}"/raft_dask*.whl)

librmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBRMM_WHEELHOUSE}"/librmm*.whl)
rmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${RMM_WHEELHOUSE}"/rmm_*.whl)

distributed-ucxx-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${DISTRIBUTED_UCXX_WHEELHOUSE}"/distributed_ucxx*.whl)
libucxx-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBUCXX_WHEELHOUSE}"/libucxx*.whl)
ucxx-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${UCXX_WHEELHOUSE}"/ucxx*.whl)
EOF
