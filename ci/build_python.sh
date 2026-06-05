#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-configure-sccache
source rapids-date-string

# fake a build from a tag
# (do this after rapids-configure-sccache, to get a cached build)
GITHUB_REF="refs/tags/v26.06.00"
export GITHUB_REF

export CMAKE_GENERATOR=Ninja

rapids-print-env

# try turning off the proxy cache
unset CONDA_CHANNEL_ALIAS

#CPP_CHANNEL=$(rapids-download-conda-from-github cpp)
CUGRAPH_COMMIT=bbe88597298aba868fe7d2afc96c5b367590ab4a
CPP_CHANNEL=$(
    RAPIDS_BUILD_WORKFLOW_NAME=build.yaml \
    rapids-get-pr-artifact cugraph 5542 cpp conda "${CUGRAPH_COMMIT}"
)

echo "--- CPP channel (${CPP_CHANNEL}) contents ---"
find "${CPP_CHANNEL}" -type f -name '*'
echo ""

RAPIDS_PACKAGE_VERSION="26.06.00"
export RAPIDS_PACKAGE_VERSION

# populates `RATTLER_CHANNELS` array and `RATTLER_ARGS` array
source rapids-rattler-channel-string

# override RATTLER_ARGS (use flexible channel priority)
RATTLER_ARGS=(
    "--experimental"
    "--no-build-id"
    "--channel-priority" "disabled"
    "--output-dir" "$RAPIDS_CONDA_BLD_OUTPUT_DIR"
)

rapids-logger "Prepending channel ${CPP_CHANNEL} to RATTLER_CHANNELS"

RATTLER_CHANNELS=("--channel" "${CPP_CHANNEL}" "${RATTLER_CHANNELS[@]}")

sccache --stop-server 2>/dev/null || true

rapids-logger "Building pylibcugraph"

# --no-build-id allows for caching with `sccache`
# more info is available at
# https://rattler.build/latest/tips_and_tricks/#using-sccache-or-ccache-with-rattler-build
rattler-build build -vvv --recipe conda/recipes/pylibcugraph \
                    "${RATTLER_ARGS[@]}" \
                    "${RATTLER_CHANNELS[@]}"

sccache --show-adv-stats
sccache --stop-server >/dev/null 2>&1 || true

rapids-logger "Building cugraph"

echo "--- rattler cache contents ---"
find ${HOME}/.cache/rattler -name '*'
echo "---"

echo "--- (before build) is there a local dask-cudf somewhere? ---"
find / -name '*dask-cudf*' || true
echo "---"

rattler-build build -vvv --recipe conda/recipes/cugraph \
                    "${RATTLER_ARGS[@]}" \
                    "${RATTLER_CHANNELS[@]}" || true

echo "--- (after build) is there a local dask-cudf somewhere? ---"
find / -name '*dask-cudf*' || true
echo ""

echo "--- trying to install dask-cudf directly ---"
conda install \
  -vvv \
  --yes \
  -c rapidsai \
  -c conda-forge \
  'dask-cudf=26.6.*'
echo "---"

exit 1

sccache --show-adv-stats
sccache --stop-server >/dev/null 2>&1 || true

# remove build_cache directory to avoid uploading the entire source tree
# tracked in https://github.com/prefix-dev/rattler-build/issues/1424
rm -rf "$RAPIDS_CONDA_BLD_OUTPUT_DIR"/build_cache

RAPIDS_PACKAGE_NAME="$(rapids-package-name conda_python cugraph --stable --cuda)"
export RAPIDS_PACKAGE_NAME
