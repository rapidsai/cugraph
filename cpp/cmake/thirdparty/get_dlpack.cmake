#=============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#=============================================================================

# dlpack is used privately by libcugraph to interpret opaque DLPack managed
# tensor pointers. In conda environments it is provided as an installed
# package; otherwise we fall back to downloading the headers.
function(find_and_configure_dlpack VERSION)

    include(${rapids-cmake-dir}/find/generate_module.cmake)
    rapids_find_generate_module(dlpack HEADER_NAMES dlpack.h)

    rapids_cpm_find(dlpack ${VERSION}
        GIT_REPOSITORY  https://github.com/dmlc/dlpack.git
        GIT_TAG         v${VERSION}
        GIT_SHALLOW     TRUE
        DOWNLOAD_ONLY   TRUE
        OPTIONS         "BUILD_MOCK OFF"
    )

    if(DEFINED dlpack_SOURCE_DIR)
        # otherwise find_package(DLPACK) will set this variable
        set(DLPACK_INCLUDE_DIR "${dlpack_SOURCE_DIR}/include" PARENT_SCOPE)
    endif()

endfunction()

set(CUGRAPH_MIN_VERSION_dlpack 1.0)

find_and_configure_dlpack(${CUGRAPH_MIN_VERSION_dlpack})
