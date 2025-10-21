# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2020-2023, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

# This function finds CCCL and sets any additional necessary environment variables.
function(find_and_configure_cccl)
  include(${rapids-cmake-dir}/cpm/cccl.cmake)
  rapids_cpm_cccl(BUILD_EXPORT_SET cugraph-exports INSTALL_EXPORT_SET cugraph-exports)
endfunction()

find_and_configure_cccl()
