# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Have cython use python 3 syntax
# cython: language_level = 3

from libc.stdint cimport (
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)

from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)
from pylibcugraph._cugraph_c.types cimport cugraph_data_type_id_t

cdef extern from "cugraph_c/dlpack_interop.h" nogil:
    ctypedef enum cugraph_dlpack_device_type_t:
        CUGRAPH_DL_DEVICE_TYPE_CPU
        CUGRAPH_DL_DEVICE_TYPE_CUDA
        CUGRAPH_DL_DEVICE_TYPE_CUDA_HOST
        CUGRAPH_DL_DEVICE_TYPE_CUDA_MANAGED

    ctypedef struct cugraph_dlpack_device_t:
        cugraph_dlpack_device_type_t device_type
        int32_t device_id

    ctypedef enum cugraph_dlpack_data_type_code_t:
        CUGRAPH_DL_DATA_TYPE_CODE_INT
        CUGRAPH_DL_DATA_TYPE_CODE_UINT
        CUGRAPH_DL_DATA_TYPE_CODE_FLOAT
        CUGRAPH_DL_DATA_TYPE_CODE_BFLOAT
        CUGRAPH_DL_DATA_TYPE_CODE_COMPLEX
        CUGRAPH_DL_DATA_TYPE_CODE_BOOL

    ctypedef struct cugraph_dlpack_data_type_t:
        uint8_t code
        uint8_t bits
        uint16_t lanes

    ctypedef struct cugraph_dlpack_tensor_t:
        void* data
        cugraph_dlpack_device_t device
        int32_t ndim
        cugraph_dlpack_data_type_t dtype
        int64_t* shape
        int64_t* strides
        uint64_t byte_offset

    ctypedef struct cugraph_dlpack_managed_tensor_t:
        cugraph_dlpack_tensor_t dl_tensor

    ctypedef struct cugraph_dlpack_version_t:
        uint32_t major
        uint32_t minor

    ctypedef struct cugraph_dlpack_managed_tensor_versioned_t:
        cugraph_dlpack_version_t version
        void* manager_ctx
        void* deleter
        uint64_t flags
        cugraph_dlpack_tensor_t dl_tensor

    cdef cugraph_error_code_t cugraph_data_type_id_from_dlpack(
        const cugraph_dlpack_data_type_t* dlpack_dtype,
        cugraph_data_type_id_t* dtype,
        cugraph_error_t** error,
    )
