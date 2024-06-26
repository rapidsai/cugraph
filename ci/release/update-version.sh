#!/bin/bash
# Copyright (c) 2018-2024, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## Usage
# bash update-version.sh <new_version>


# Format is YY.MM.PP - no leading 'v' or trailing 'a'
NEXT_FULL_TAG=$1

# Get current version
CURRENT_TAG=$(git tag --merged HEAD | grep -xE '^v.*' | sort --version-sort | tail -n 1 | tr -d 'v')
CURRENT_MAJOR=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[1]}')
CURRENT_MINOR=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[2]}')
CURRENT_PATCH=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[3]}')
CURRENT_SHORT_TAG=${CURRENT_MAJOR}.${CURRENT_MINOR}

#Get <major>.<minor> for next version
NEXT_MAJOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[2]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}
NEXT_UCX_PY_VERSION="$(curl -sL https://version.gpuci.io/rapids/${NEXT_SHORT_TAG})"

echo "Preparing release $CURRENT_TAG => $NEXT_FULL_TAG"

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' $2 && rm -f ${2}.bak
}

# Centralized version file update
# NOTE: Any script that runs in CI will need to use gha-tool `rapids-generate-version`
# and echo it to `VERSION` file to get an alpha spec of the current version
echo "${NEXT_FULL_TAG}" > VERSION

# Need to distutils-normalize the original version
NEXT_SHORT_TAG_PEP440=$(python -c "from setuptools.extern import packaging; print(packaging.version.Version('${NEXT_SHORT_TAG}'))")

DEPENDENCIES=(
  cudf
  cugraph
  cugraph-dgl
  cugraph-pyg
  cugraph-service-server
  cugraph-service-client
  cuxfilter
  dask-cuda
  dask-cudf
  libcudf
  libcugraphops
  libraft
  libraft-headers
  librmm
  pylibcugraph
  pylibcugraphops
  pylibwholegraph
  pylibraft
  pyraft
  raft-dask
  rmm
  ucx-py
  rapids-dask-dependency
)
for DEP in "${DEPENDENCIES[@]}"; do
  for FILE in dependencies.yaml conda/environments/*.yaml python/cugraph-{pyg,dgl}/conda/*.yaml; do
    sed_runner "/-.* ${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==/ s/==.*/==${NEXT_SHORT_TAG_PEP440}.*/g" "${FILE}"
    sed_runner "/-.* ucx-py==/ s/==.*/==${NEXT_UCX_PY_VERSION}.*/g" "${FILE}"
  done
  for FILE in python/**/pyproject.toml python/**/**/pyproject.toml; do
    sed_runner "/\"${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==/ s/==.*\"/==${NEXT_SHORT_TAG_PEP440}.*\"/g" "${FILE}"
    sed_runner "/\"ucx-py==/ s/==.*\"/==${NEXT_UCX_PY_VERSION}.*\"/g" "${FILE}"
  done
done

# ucx-py version
sed_runner "/^ucx_py_version:$/ {n;s/.*/  - \"${NEXT_UCX_PY_VERSION}.*\"/}" conda/recipes/cugraph/conda_build_config.yaml
sed_runner "/^ucx_py_version:$/ {n;s/.*/  - \"${NEXT_UCX_PY_VERSION}.*\"/}" conda/recipes/cugraph-service/conda_build_config.yaml
sed_runner "/^ucx_py_version:$/ {n;s/.*/  - \"${NEXT_UCX_PY_VERSION}.*\"/}" conda/recipes/pylibcugraph/conda_build_config.yaml

# CI files
for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-workflows/ s/@.*/@branch-${NEXT_SHORT_TAG}/g" "${FILE}"
  # Wheel builds install dask-cuda from source, update its branch
  sed_runner "s/dask-cuda.git@branch-[0-9][0-9].[0-9][0-9]/dask-cuda.git@branch-${NEXT_SHORT_TAG}/g" "${FILE}"
done

sed_runner "s/branch-.*/branch-${NEXT_SHORT_TAG}/g" python/nx-cugraph/README.md

# .devcontainer files
find .devcontainer/ -type f -name devcontainer.json -print0 | while IFS= read -r -d '' filename; do
    sed_runner "s@rapidsai/devcontainers:[0-9.]*@rapidsai/devcontainers:${NEXT_SHORT_TAG}@g" "${filename}"
    sed_runner "s@rapidsai/devcontainers/features/ucx:[0-9.]*@rapidsai/devcontainers/features/ucx:${NEXT_SHORT_TAG_PEP440}@" "${filename}"
    sed_runner "s@rapidsai/devcontainers/features/cuda:[0-9.]*@rapidsai/devcontainers/features/cuda:${NEXT_SHORT_TAG_PEP440}@" "${filename}"
    sed_runner "s@rapidsai/devcontainers/features/rapids-build-utils:[0-9.]*@rapidsai/devcontainers/features/rapids-build-utils:${NEXT_SHORT_TAG_PEP440}@" "${filename}"
    sed_runner "s@rapids-\${localWorkspaceFolderBasename}-[0-9.]*@rapids-\${localWorkspaceFolderBasename}-${NEXT_SHORT_TAG}@g" "${filename}"
done

sed_runner "s/:[0-9][0-9]\.[0-9][0-9]/:${NEXT_SHORT_TAG}/" ./notebooks/README.md
sed_runner "s/branch-[0-9][0-9].[0-9][0-9]/branch-${NEXT_SHORT_TAG}/" ./docs/cugraph/source/nx_cugraph/nx_cugraph.md
