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

cdef extern from "dlpack/dlpack.h" nogil:
    ctypedef enum DLDeviceType:
        kDLCPU
        kDLCUDA
        kDLCUDAHost
        kDLCUDAManaged

    ctypedef struct DLDevice:
        DLDeviceType device_type
        int32_t device_id

    cdef enum DLDataTypeCode:
        kDLInt
        kDLUInt
        kDLFloat
        kDLBfloat
        kDLComplex
        kDLBool

    ctypedef struct DLDataType:
        uint8_t code
        uint8_t bits
        uint16_t lanes

    ctypedef struct DLTensor:
        void* data
        DLDevice device
        int32_t ndim
        DLDataType dtype
        int64_t* shape
        int64_t* strides
        uint64_t byte_offset

    ctypedef struct DLManagedTensor:
        DLTensor dl_tensor


# pylibcugraph currently builds against DLPack 0.8, which predates the
# versioned managed-tensor declaration. Mirror the DLPack 1.0 layout here so
# newer producers can communicate flags such as read-only without requiring a
# project-wide DLPack dependency upgrade. manager_ctx and deleter are retained
# as opaque pointers because only their size and position affect the ABI here.
ctypedef struct pylibcugraph_DLPackVersion:
    uint32_t major
    uint32_t minor

ctypedef struct pylibcugraph_DLManagedTensorVersioned:
    pylibcugraph_DLPackVersion version
    void* manager_ctx
    void* deleter
    uint64_t flags
    DLTensor dl_tensor

cdef extern from "cugraph_c/dlpack_interop.h":
    cdef cugraph_error_code_t cugraph_data_type_id_from_dlpack(
        const DLDataType* dlpack_dtype,
        cugraph_data_type_id_t* dtype,
        cugraph_error_t** error,
    )
