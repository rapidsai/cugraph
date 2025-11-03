#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

function(find_and_configure_nccl)

    if(TARGET NCCL::NCCL)
        return()
    endif()

    rapids_find_generate_module(NCCL
        HEADER_NAMES  nccl.h
        LIBRARY_NAMES nccl
    )

    # Currently NCCL has no CMake build-system so we require
    # it built and installed on the machine already
    rapids_find_package(NCCL REQUIRED)

endfunction()

find_and_configure_nccl()
