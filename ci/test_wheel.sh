#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -eoxu pipefail

package_name=$1

# Run smoke tests for aarch64 pull requests
arch=$(uname -m)
if [[ "${arch}" == "aarch64" && ${RAPIDS_BUILD_TYPE} == "pull-request" ]]; then
    python ./ci/wheel_smoke_test_"${package_name}".py
# FIXME: temporarily disables MG tests due to hang in CI with CUDA 13.1.0
#else
#    python_package_name=${package_name//-/_}
#    # Test runs that include tests that use dask require
#    # --import-mode=append. See test_python.sh for details.
#    # FIXME: Adding PY_IGNORE_IMPORTMISMATCH=1 to workaround conftest.py import
#    # mismatch error seen by nx-cugraph after using pytest 8 and
#    # --import-mode=append.
#    RAPIDS_DATASET_ROOT_DIR=$(pwd)/datasets \
#    PY_IGNORE_IMPORTMISMATCH=1 \
#    DASK_WORKER_DEVICES="0" \
#    DASK_DISTRIBUTED__SCHEDULER__WORKER_TTL="1000s" \
#    DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT="1000s" \
#    DASK_CUDA_WAIT_WORKERS_MIN_TIMEOUT="1000s" \
#    python -m pytest \
#       -v \
#       --import-mode=append \
#       --benchmark-disable \
#       "./python/${package_name}/${python_package_name}/tests"
fi
