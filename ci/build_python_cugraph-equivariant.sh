#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

rapids-print-env

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

version=$(rapids-generate-version)
git_commit=$(git rev-parse HEAD)
export RAPIDS_PACKAGE_VERSION=${version}
echo "${version}" > VERSION

rapids-logger "Begin py build"

package_dir="python"
for package_name in cugraph-equivariant; do
  underscore_package_name=$(echo "${package_name}" | tr "-" "_")
  sed -i "/^__git_commit__/ s/= .*/= \"${git_commit}\"/g" "${package_dir}/${package_name}/${underscore_package_name}/_version.py"
done

rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
  conda/recipes/cugraph-equivariant

rapids-upload-conda-to-s3 python