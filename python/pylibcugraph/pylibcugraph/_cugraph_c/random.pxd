# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Have cython use python 3 syntax
# cython: language_level = 3

from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)
from pylibcugraph._cugraph_c.resource_handle cimport (
    cugraph_resource_handle_t,
)

cdef extern from "cugraph_c/random.h":
    ctypedef struct cugraph_rng_state_t:
        pass

    cdef cugraph_error_code_t cugraph_rng_state_create(
        const cugraph_resource_handle_t* handle,
        size_t seed,
        cugraph_rng_state_t** state,
        cugraph_error_t** error,
    )

    cdef void cugraph_rng_state_free(cugraph_rng_state_t* p)
