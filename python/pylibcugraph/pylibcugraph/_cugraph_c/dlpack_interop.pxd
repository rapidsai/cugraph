# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Have cython use python 3 syntax
# cython: language_level = 3

from libc.stdint cimport uint8_t, uint16_t

from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)
from pylibcugraph._cugraph_c.types cimport cugraph_data_type_id_t

cdef extern from "dlpack/dlpack.h" nogil:
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

cdef extern from "cugraph_c/dlpack_interop.h":
    cdef cugraph_error_code_t cugraph_data_type_id_from_dlpack(
        const DLDataType* dlpack_dtype,
        cugraph_data_type_id_t* dtype,
        cugraph_error_t** error,
    )
