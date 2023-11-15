#!/bin/bash
# Copyright (c) 2018-2023, NVIDIA CORPORATION.
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

# rapids-cmake version
sed_runner 's/'"branch-.*\/RAPIDS.cmake"'/'"branch-${NEXT_SHORT_TAG}\/RAPIDS.cmake"'/g' fetch_rapids.cmake

# CMakeLists update
sed_runner 's/'"CUGRAPH VERSION .* LANGUAGES C CXX CUDA)"'/'"CUGRAPH VERSION ${NEXT_FULL_TAG} LANGUAGES C CXX CUDA)"'/g' cpp/CMakeLists.txt
sed_runner 's|'"branch-.*/RAPIDS.cmake"'|'"branch-${NEXT_SHORT_TAG}/RAPIDS.cmake"'|g' cpp/CMakeLists.txt
sed_runner 's/'"CUGRAPH_ETL VERSION .* LANGUAGES C CXX CUDA)"'/'"CUGRAPH_ETL VERSION ${NEXT_FULL_TAG} LANGUAGES C CXX CUDA)"'/g' cpp/libcugraph_etl/CMakeLists.txt
sed_runner 's|'"branch-.*/RAPIDS.cmake"'|'"branch-${NEXT_SHORT_TAG}/RAPIDS.cmake"'|g' cpp/libcugraph_etl/CMakeLists.txt
sed_runner "s/set(pylibcugraph_version .*)/set(pylibcugraph_version ${NEXT_FULL_TAG})/g" python/pylibcugraph/CMakeLists.txt
sed_runner "s/set(cugraph_version .*)/set(cugraph_version ${NEXT_FULL_TAG})/g" python/cugraph/CMakeLists.txt

# RTD update
sed_runner 's/version = .*/version = '"'${NEXT_SHORT_TAG}'"'/g' docs/cugraph/source/conf.py
sed_runner 's/release = .*/release = '"'${NEXT_FULL_TAG}'"'/g' docs/cugraph/source/conf.py

# Centralized version file update
# NOTE: Any script that runs in CI will need to use gha-tool `rapids-generate-version`
# and echo it to `VERSION` file to get an alpha spec of the current version
echo "${NEXT_FULL_TAG}" > VERSION

# Wheel testing script
sed_runner "s/branch-.*/branch-${NEXT_SHORT_TAG}/g" ci/test_wheel_cugraph.sh

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
    sed_runner "/-.* ${DEP}==/ s/==.*/==${NEXT_SHORT_TAG_PEP440}.*/g" ${FILE}
    sed_runner "/-.* ${DEP}-cu[0-9][0-9]==/ s/==.*/==${NEXT_SHORT_TAG_PEP440}.*/g" ${FILE}
    sed_runner "/-.* ucx-py==/ s/==.*/==${NEXT_UCX_PY_VERSION}.*/g" ${FILE}
  done
  for FILE in python/**/pyproject.toml python/**/**/pyproject.toml; do
    sed_runner "/\"${DEP}==/ s/==.*\"/==${NEXT_SHORT_TAG_PEP440}.*\"/g" ${FILE}
    sed_runner "/\"ucx-py==/ s/==.*\"/==${NEXT_UCX_PY_VERSION}.*\"/g" ${FILE}
  done
done

# Doxyfile update
sed_runner "s|PROJECT_NUMBER[[:space:]]*=.*|PROJECT_NUMBER=${NEXT_SHORT_TAG}|" cpp/doxygen/Doxyfile

# ucx-py version
sed_runner "/^ucx_py_version:$/ {n;s/.*/  - \"${NEXT_UCX_PY_VERSION}.*\"/}" conda/recipes/cugraph/conda_build_config.yaml
sed_runner "/^ucx_py_version:$/ {n;s/.*/  - \"${NEXT_UCX_PY_VERSION}.*\"/}" conda/recipes/cugraph-service/conda_build_config.yaml
sed_runner "/^ucx_py_version:$/ {n;s/.*/  - \"${NEXT_UCX_PY_VERSION}.*\"/}" conda/recipes/pylibcugraph/conda_build_config.yaml

# nx-cugraph NetworkX entry-point meta-data
sed_runner "s@branch-[0-9][0-9].[0-9][0-9]@branch-${NEXT_SHORT_TAG}@g" python/nx-cugraph/_nx_cugraph/__init__.py
# FIXME: can this use the standard VERSION file and update mechanism?
sed_runner "s/__version__ = .*/__version__ = \"${NEXT_FULL_TAG}\"/g" python/nx-cugraph/_nx_cugraph/__init__.py

# CI files
for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-workflows/ s/@.*/@branch-${NEXT_SHORT_TAG}/g" "${FILE}"
  # Wheel builds clone cugraph-ops, update its branch
  sed_runner "s/extra-repo-sha: branch-.*/extra-repo-sha: branch-${NEXT_SHORT_TAG}/g" "${FILE}"
  # Wheel builds install dask-cuda from source, update its branch
  sed_runner "s/dask-cuda.git@branch-[0-9][0-9].[0-9][0-9]/dask-cuda.git@branch-${NEXT_SHORT_TAG}/g" "${FILE}"
done
sed_runner "s/RAPIDS_VERSION_NUMBER=\".*/RAPIDS_VERSION_NUMBER=\"${NEXT_SHORT_TAG}\"/g" ci/build_docs.sh

sed_runner "s/branch-.*/branch-${NEXT_SHORT_TAG}/g" python/nx-cugraph/README.md

# .devcontainer files
find .devcontainer/ -type f -name devcontainer.json -print0 | while IFS= read -r -d '' filename; do
    sed_runner "s@rapidsai/devcontainers:[0-9.]*@rapidsai/devcontainers:${NEXT_SHORT_TAG}@g" "${filename}"
    sed_runner "s@rapidsai/devcontainers/features/ucx:[0-9.]*@rapidsai/devcontainers/features/ucx:${NEXT_SHORT_TAG_PEP440}@" "${filename}"
    sed_runner "s@rapidsai/devcontainers/features/rapids-build-utils:[0-9.]*@rapidsai/devcontainers/features/rapids-build-utils:${NEXT_SHORT_TAG_PEP440}@" "${filename}"
done
