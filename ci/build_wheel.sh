#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

package_name=$1
package_dir=$2

source rapids-configure-sccache
source rapids-date-string

# Use gha-tools rapids-pip-wheel-version to generate wheel version then
# update the necessary files
version_override="$(rapids-pip-wheel-version ${RAPIDS_DATE_STRING})"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

bash ci/release/apply_wheel_modifications.sh ${version_override} "-${RAPIDS_PY_CUDA_SUFFIX}"
echo "The package name and/or version was modified in the package source. The git diff is:"
git diff

# Install CI tools using pip
python -m pip install "rapids-dependency-file-generator==1.*"

# Temporary patch for adding it to the PATH
pyenv rehash

# Manually install dependencies because we're building without isolation.
rapids-dependency-file-generator \
  --output requirements \
  --file_key py_build_${package_name} \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee requirements.txt
python -m pip install -r requirements.txt

cd "${package_dir}"

python -m pip wheel . -w dist -vvv --no-deps --disable-pip-version-check --no-build-isolation

mkdir -p final_dist
python -m auditwheel repair -w final_dist dist/*

RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 final_dist
