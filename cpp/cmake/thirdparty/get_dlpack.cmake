#=============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#=============================================================================

# dlpack is a header-only library used by the cugraph_c DLPack interop API.
# In conda environments it is provided as an installed CMake package; in other
# environments (e.g. pip / devcontainer builds) it may not be installed, so we
# fall back to downloading the headers via CPM.
function(find_and_configure_dlpack VERSION)

    rapids_cpm_find(dlpack ${VERSION}
        GLOBAL_TARGETS  dlpack::dlpack
        CPM_ARGS
            GIT_REPOSITORY  https://github.com/dmlc/dlpack.git
            GIT_TAG         v${VERSION}
            GIT_SHALLOW     TRUE
            DOWNLOAD_ONLY   TRUE
            OPTIONS         "BUILD_MOCK OFF"
    )

    if(DEFINED dlpack_SOURCE_DIR)
        # dlpack was downloaded from source (no installed CMake package was
        # found). It ships no usable CMake config in DOWNLOAD_ONLY mode, so
        # expose an interface target matching the one from an installed
        # dlpackConfig.cmake. The include directory is build-tree only; installed
        # consumers obtain dlpack.h from their own environment (e.g. conda).
        set(DLPACK_INCLUDE_DIRS "${dlpack_SOURCE_DIR}/include" PARENT_SCOPE)
        if(NOT TARGET dlpack::dlpack)
            add_library(dlpack::dlpack INTERFACE IMPORTED GLOBAL)
            target_include_directories(dlpack::dlpack INTERFACE
                "$<BUILD_INTERFACE:${dlpack_SOURCE_DIR}/include>")
        endif()
    elseif(TARGET dlpack::dlpack)
        get_target_property(_dlpack_include_dirs dlpack::dlpack
            INTERFACE_INCLUDE_DIRECTORIES)
        set(DLPACK_INCLUDE_DIRS "${_dlpack_include_dirs}" PARENT_SCOPE)
    endif()

endfunction()

set(CUGRAPH_MIN_VERSION_dlpack 0.8)

find_and_configure_dlpack(${CUGRAPH_MIN_VERSION_dlpack})
