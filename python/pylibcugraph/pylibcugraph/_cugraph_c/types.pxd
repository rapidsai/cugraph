# SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Have cython use python 3 syntax
# cython: language_level = 3

from libc.stdint cimport int8_t


cdef extern from "cugraph_c/types.h":

    ctypedef enum bool_t:
        FALSE
        TRUE

    ctypedef enum cugraph_data_type_id_t:
        INT8
        INT16
        INT32
        INT64
        UINT8
        UINT16
        UINT32
        UINT64
        FLOAT32
        FLOAT64
        SIZE_T
        BOOL

    ctypedef int8_t byte_t
