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
NEXT_UCX_PY_VERSION="$(curl -sL https://version.gpuci.io/rapids/${NEXT_SHORT_TAG}).*"

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

for FILE in conda/environments/*.yaml dependencies.yaml; do
   sed_runner "s/libcugraphops=${CURRENT_SHORT_TAG}/libcugraphops=${NEXT_SHORT_TAG}/g" ${FILE};
   sed_runner "s/cudf=${CURRENT_SHORT_TAG}/cudf=${NEXT_SHORT_TAG}/g" ${FILE};
   sed_runner "s/rmm=${CURRENT_SHORT_TAG}/rmm=${NEXT_SHORT_TAG}/g" ${FILE};
   sed_runner "s/libraft-headers=${CURRENT_SHORT_TAG}/libraft-headers=${NEXT_SHORT_TAG}/g" ${FILE};
   sed_runner "s/pyraft=${CURRENT_SHORT_TAG}/pyraft=${NEXT_SHORT_TAG}/g" ${FILE};
   sed_runner "s/raft-dask=${CURRENT_SHORT_TAG}/raft-dask=${NEXT_SHORT_TAG}/g" ${FILE};
   sed_runner "s/pylibraft=${CURRENT_SHORT_TAG}/pylibraft=${NEXT_SHORT_TAG}/g" ${FILE};
   sed_runner "s/dask-cuda=${CURRENT_SHORT_TAG}/dask-cuda=${NEXT_SHORT_TAG}/g" ${FILE};
   sed_runner "s/dask-cudf=${CURRENT_SHORT_TAG}/dask-cudf=${NEXT_SHORT_TAG}/g" ${FILE};
   sed_runner "s/cuxfilter=${CURRENT_SHORT_TAG}/cuxfilter=${NEXT_SHORT_TAG}/g" ${FILE};
   sed_runner "s/ucx-py=.*/ucx-py=${NEXT_UCX_PY_VERSION}/g" ${FILE};
done

# Doxyfile update
sed_runner "s|PROJECT_NUMBER[[:space:]]*=.*|PROJECT_NUMBER=${NEXT_SHORT_TAG}|" cpp/doxygen/Doxyfile

# ucx-py version
sed_runner "s/export UCX_PY_VERSION=.*/export UCX_PY_VERSION='${NEXT_UCX_PY_VERSION}'/g" ci/benchmark/build.sh
sed_runner "s/export UCX_PY_VERSION=.*/export UCX_PY_VERSION='${NEXT_UCX_PY_VERSION}'/g" ci/gpu/build.sh
sed_runner "s/export UCX_PY_VERSION=.*/export UCX_PY_VERSION='${NEXT_UCX_PY_VERSION}'/g" ci/cpu/build.sh
sed_runner "/^ucx_py_version:$/ {n;s/.*/  - \"${NEXT_UCX_PY_VERSION}\"/}" conda/recipes/cugraph/conda_build_config.yaml
sed_runner "/^ucx_py_version:$/ {n;s/.*/  - \"${NEXT_UCX_PY_VERSION}\"/}" conda/recipes/cugraph-service/conda_build_config.yaml
sed_runner "/^ucx_py_version:$/ {n;s/.*/  - \"${NEXT_UCX_PY_VERSION}\"/}" conda/recipes/pylibcugraph/conda_build_config.yaml

for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-action-workflows/ s/@.*/@branch-${NEXT_SHORT_TAG}/g" "${FILE}"
done

# Wheel builds clone cugraph-ops, update its branch
sed_runner "s/extra-repo-sha: branch-.*/extra-repo-sha: branch-${NEXT_SHORT_TAG}/g" .github/workflows/*.yaml

# Wheel builds install dask-cuda from source, update its branch
sed_runner "s/dask-cuda.git@branch-[^\"\s]\+/dask-cuda.git@branch-${NEXT_SHORT_TAG}/g" .github/workflows/*.yaml

# Need to distutils-normalize the original version
NEXT_SHORT_TAG_PEP440=$(python -c "from setuptools.extern import packaging; print(packaging.version.Version('${NEXT_SHORT_TAG}'))")

# Wheel builds install intra-RAPIDS dependencies from same release
sed_runner "s/{cuda_suffix}[^\"].*\",/{cuda_suffix}==${NEXT_SHORT_TAG_PEP440}.*\",/g" python/pylibcugraph/setup.py
sed_runner "s/{cuda_suffix}.*\",/{cuda_suffix}==${NEXT_SHORT_TAG_PEP440}.*\",/g" python/pylibcugraph/_custom_build/backend.py
sed_runner "s/{cuda_suffix}[^\"].*\",/{cuda_suffix}==${NEXT_SHORT_TAG_PEP440}.*\",/g" python/cugraph/setup.py
sed_runner "s/{cuda_suffix}.*\",/{cuda_suffix}==${NEXT_SHORT_TAG_PEP440}.*\",/g" python/cugraph/_custom_build/backend.py
