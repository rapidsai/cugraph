#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-configure-sccache
source rapids-datetime-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-generate-version > ./VERSION

RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION)
export RAPIDS_PACKAGE_VERSION

# populates `RATTLER_CHANNELS` array and `RATTLER_ARGS` array
source rapids-rattler-channel-string

# A skipped 'conda-cpp-build' job (libcugraph unaffected by this PR) means no
# artifact was uploaded this run. Fail fast (short retry budget) rather than
# waiting out the full retry window, then fall back to the recipe's own version
# constraint resolving libcugraph from the rapidsai-nightly channel already
# present in RATTLER_CHANNELS.
if CPP_CHANNEL=$(RAPIDS_RETRY_MAX=1 RAPIDS_RETRY_SLEEP=15 rapids-download-from-github "$(rapids-artifact-name conda_cpp libcugraph cugraph --cuda "$RAPIDS_CUDA_VERSION")" 2>/tmp/libcugraph_channel_download.log); then
  rapids-logger "Prepending channel ${CPP_CHANNEL} to RATTLER_CHANNELS"

  RATTLER_CHANNELS=("--channel" "${CPP_CHANNEL}" "${RATTLER_CHANNELS[@]}")
else
  cat /tmp/libcugraph_channel_download.log >&2
  rapids-logger "No libcugraph artifact found for this run; resolving it from the nightly channel"
fi

sccache --stop-server 2>/dev/null || true

rapids-logger "Building pylibcugraph"

# --no-build-id allows for caching with `sccache`
# more info is available at
# https://rattler.build/latest/tips_and_tricks/#using-sccache-or-ccache-with-rattler-build
rattler-build build --recipe conda/recipes/pylibcugraph \
                    "${RATTLER_ARGS[@]}" \
                    "${RATTLER_CHANNELS[@]}"

sccache --show-adv-stats
sccache --stop-server >/dev/null 2>&1 || true

rapids-logger "Building cugraph"

rattler-build build --recipe conda/recipes/cugraph \
                    "${RATTLER_ARGS[@]}" \
                    "${RATTLER_CHANNELS[@]}"

sccache --show-adv-stats
sccache --stop-server >/dev/null 2>&1 || true

# remove build_cache directory to avoid uploading the entire source tree
# tracked in https://github.com/prefix-dev/rattler-build/issues/1424
rm -rf "$RAPIDS_CONDA_BLD_OUTPUT_DIR"/build_cache

RAPIDS_PACKAGE_NAME="$(rapids-artifact-name conda_python cugraph cugraph --stable --cuda "$RAPIDS_CUDA_VERSION")"
export RAPIDS_PACKAGE_NAME
