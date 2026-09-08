# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Have cython use python 3 syntax
# cython: language_level = 3

from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)
from pylibcugraph._cugraph_c.types cimport (
    bool_t,
    cugraph_data_type_id_t,
)

cdef extern from "cugraph_c/dlpack_interop.h" nogil:
    cdef cugraph_error_code_t cugraph_dlpack_is_device_accessible(
        const void* managed_tensor,
        bool_t versioned,
        bool_t* result,
        cugraph_error_t** error,
    )

    cdef cugraph_error_code_t cugraph_dlpack_is_host_accessible(
        const void* managed_tensor,
        bool_t versioned,
        bool_t* result,
        cugraph_error_t** error,
    )

    cdef cugraph_error_code_t cugraph_dlpack_get_array_info(
        const void* managed_tensor,
        bool_t versioned,
        void** data,
        size_t* size,
        cugraph_data_type_id_t* dtype,
        cugraph_error_t** error,
    )
