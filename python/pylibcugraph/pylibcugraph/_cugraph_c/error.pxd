# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Have cython use python 3 syntax
# cython: language_level = 3

cdef extern from "cugraph_c/error.h":

    ctypedef enum cugraph_error_code_t:
        CUGRAPH_SUCCESS
        CUGRAPH_UNKNOWN_ERROR
        CUGRAPH_INVALID_HANDLE
        CUGRAPH_ALLOC_ERROR
        CUGRAPH_INVALID_INPUT
        CUGRAPH_NOT_IMPLEMENTED
        CUGRAPH_UNSUPPORTED_TYPE_COMBINATION

    ctypedef struct cugraph_error_t:
       pass

    const char* \
        cugraph_error_message(
            const cugraph_error_t* error
        )

    void \
        cugraph_error_free(
            cugraph_error_t* error
        )
