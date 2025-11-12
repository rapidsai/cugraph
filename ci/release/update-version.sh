#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

## Usage
# Primary interface: ./ci/release/update-version.sh --run-context=main|release <new_version>
# Fallback: Environment variable support for automation needs
# NOTE: Must be run from the root of the repository
#
# CLI args take precedence when both are provided
# If neither RUN_CONTEXT nor --run-context is provided, defaults to main
#
# Examples:
#   ./ci/release/update-version.sh --run-context=main 25.12.00
#   ./ci/release/update-version.sh --run-context=release 25.12.00
#   RAPIDS_RUN_CONTEXT=main ./ci/release/update-version.sh 25.12.00

# Parse command line arguments
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --run-context=*)
      CLI_RUN_CONTEXT="${1#*=}"
      shift
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

# Restore positional parameters
set -- "${POSITIONAL_ARGS[@]}"

# Determine RUN_CONTEXT with precedence: CLI > Environment > Default
if [[ -n "${CLI_RUN_CONTEXT:-}" ]]; then
    RUN_CONTEXT="${CLI_RUN_CONTEXT}"
    echo "Using run-context from CLI: ${RUN_CONTEXT}"
elif [[ -n "${RAPIDS_RUN_CONTEXT:-}" ]]; then
    RUN_CONTEXT="${RAPIDS_RUN_CONTEXT}"
    echo "Using RUN_CONTEXT from environment: ${RUN_CONTEXT}"
else
    RUN_CONTEXT="main"
    echo "Using default run-context: ${RUN_CONTEXT}"
fi

# Validate RUN_CONTEXT
if [[ "${RUN_CONTEXT}" != "main" && "${RUN_CONTEXT}" != "release" ]]; then
    echo "Error: Invalid run-context '${RUN_CONTEXT}'. Must be 'main' or 'release'"
    exit 1
fi

# Format is YY.MM.PP - no leading 'v' or trailing 'a'
NEXT_FULL_TAG=$1

# Get current version
CURRENT_TAG=$(git tag --merged HEAD | grep -xE '^v.*' | sort --version-sort | tail -n 1 | tr -d 'v')

#Get <major>.<minor> for next version
NEXT_MAJOR=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[2]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}
NEXT_UCXX_SHORT_TAG="$(curl -sL https://version.gpuci.io/rapids/"${NEXT_SHORT_TAG}")"

# Determine branch name based on context
if [[ "${RUN_CONTEXT}" == "main" ]]; then
    RAPIDS_BRANCH_NAME="main"
    echo "Preparing development branch update ${CURRENT_TAG} => ${NEXT_FULL_TAG} (targeting main branch)"
elif [[ "${RUN_CONTEXT}" == "release" ]]; then
    RAPIDS_BRANCH_NAME="release/${NEXT_SHORT_TAG}"
    echo "Preparing release branch update ${CURRENT_TAG} => ${NEXT_FULL_TAG} (targeting release/${NEXT_SHORT_TAG} branch)"
fi

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' "$2" && rm -f "${2}".bak
}

# Centralized version file update
# NOTE: Any script that runs in CI will need to use gha-tool `rapids-generate-version`
# and echo it to `VERSION` file to get an alpha spec of the current version
echo "${NEXT_FULL_TAG}" > VERSION
echo "${RAPIDS_BRANCH_NAME}" > RAPIDS_BRANCH

# Need to distutils-normalize the original version
NEXT_SHORT_TAG_PEP440=$(python -c "from packaging.version import Version; print(Version('${NEXT_SHORT_TAG}'))")
NEXT_UCXX_SHORT_TAG_PEP440=$(python -c "from packaging.version import Version; print(Version('${NEXT_UCXX_SHORT_TAG}'))")

DEPENDENCIES=(
  cudf
  cugraph
  cuxfilter
  dask-cuda
  dask-cudf
  libcudf
  libcugraph
  libcugraph_etl
  libcugraph-tests
  libraft
  librmm
  pylibcudf
  pylibcugraph
  pylibraft
  pyraft
  raft-dask
  rmm
  rapids-dask-dependency
)
UCXX_DEPENDENCIES=(
  libucxx
  ucxx
)
for FILE in dependencies.yaml conda/environments/*.yaml; do
  for DEP in "${DEPENDENCIES[@]}"; do
    sed_runner "/-.* ${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}\(\[.*\]\)\{0,1\}==/ s/==.*/==${NEXT_SHORT_TAG_PEP440}.*,>=0.0.0a0/g" "${FILE}"
  done
  for DEP in "${UCXX_DEPENDENCIES[@]}"; do
    sed_runner "/-.* ${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==/ s/==.*/==${NEXT_UCXX_SHORT_TAG_PEP440}.*,>=0.0.0a0/g" "${FILE}"
  done
done
for FILE in python/*/pyproject.toml; do
  for DEP in "${DEPENDENCIES[@]}"; do
    sed_runner "/\"${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==/ s/==.*\"/==${NEXT_SHORT_TAG_PEP440}.*,>=0.0.0a0\"/g" "${FILE}"
  done
  for DEP in "${UCXX_DEPENDENCIES[@]}"; do
    sed_runner "/\"${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==/ s/==.*\"/==${NEXT_UCXX_SHORT_TAG_PEP440}.*,>=0.0.0a0\"/g" "${FILE}"
  done
done

# ucxx version
for FILE in conda/recipes/*/conda_build_config.yaml; do
  sed_runner "/^libucxx_version:\$/ {n;s|.*|  - \"${NEXT_UCXX_SHORT_TAG_PEP440}.*\"|;}" "${FILE}"
  sed_runner "/^ucxx_version:\$/ {n;s|.*|  - \"${NEXT_UCXX_SHORT_TAG_PEP440}.*\"|;}" "${FILE}"
done

# CI files - context-aware branch references
for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-workflows/ s|@.*|@${RAPIDS_BRANCH_NAME}|g" "${FILE}"
  # Wheel builds install dask-cuda from source, update its branch (context-aware)
  if [[ "${RUN_CONTEXT}" == "main" ]]; then
    sed_runner "s|dask-cuda.git@release/[0-9][0-9].[0-9][0-9]|dask-cuda.git@main|g" "${FILE}"
  elif [[ "${RUN_CONTEXT}" == "release" ]]; then
    sed_runner "s|dask-cuda.git@main|dask-cuda.git@release/${NEXT_SHORT_TAG}|g" "${FILE}"
  fi
  sed_runner "s/:[0-9]*\\.[0-9]*-/:${NEXT_SHORT_TAG}-/g" "${FILE}"
done

# Documentation references - context-aware  
if [[ "${RUN_CONTEXT}" == "main" ]]; then
  # In main context, keep documentation on main (no changes needed)
  :
elif [[ "${RUN_CONTEXT}" == "release" ]]; then
  # In release context, use release branch for documentation links (word boundaries to avoid partial matches)
  sed_runner "s|\\bmain\\b|release/${NEXT_SHORT_TAG}|g" readme_pages/CONTRIBUTING.md
  sed_runner "s|\\bmain\\b|release/${NEXT_SHORT_TAG}|g" notebooks/modules/mag240m_pg.ipynb
  sed_runner "s|\\bmain\\b|release/${NEXT_SHORT_TAG}|g" notebooks/demo/mg_property_graph.ipynb
  sed_runner "s|\\bmain\\b|release/${NEXT_SHORT_TAG}|g" notebooks/README.md
fi

# .devcontainer files
find .devcontainer/ -type f -name devcontainer.json -print0 | while IFS= read -r -d '' filename; do
    sed_runner "s@rapidsai/devcontainers:[0-9.]*@rapidsai/devcontainers:${NEXT_SHORT_TAG}@g" "${filename}"
    sed_runner "s@rapidsai/devcontainers/features/ucx:[0-9.]*@rapidsai/devcontainers/features/ucx:${NEXT_SHORT_TAG_PEP440}@" "${filename}"
    sed_runner "s@rapidsai/devcontainers/features/cuda:[0-9.]*@rapidsai/devcontainers/features/cuda:${NEXT_SHORT_TAG_PEP440}@" "${filename}"
    sed_runner "s@rapidsai/devcontainers/features/rapids-build-utils:[0-9.]*@rapidsai/devcontainers/features/rapids-build-utils:${NEXT_SHORT_TAG_PEP440}@" "${filename}"
    sed_runner "s@rapids-\${localWorkspaceFolderBasename}-[0-9.]*@rapids-\${localWorkspaceFolderBasename}-${NEXT_SHORT_TAG}@g" "${filename}"
done

sed_runner "s/:[0-9][0-9]\.[0-9][0-9]/:${NEXT_SHORT_TAG}/" ./notebooks/README.md
