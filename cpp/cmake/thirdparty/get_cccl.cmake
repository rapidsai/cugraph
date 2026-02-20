# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2020-2023, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

# This function finds CCCL and sets any additional necessary environment variables.
function(find_and_configure_cccl)
  include(${rapids-cmake-dir}/cpm/cccl.cmake)
  include(${rapids-cmake-dir}/cpm/package_override.cmake)
  rapids_cpm_package_override("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/cccl_override.json")
  set(CCCL_ENABLE_UNSTABLE ON)
  rapids_cpm_cccl(BUILD_EXPORT_SET cugraph-exports INSTALL_EXPORT_SET cugraph-exports)
endfunction()

find_and_configure_cccl()
