# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Have cython use python 3 syntax
# cython: language_level = 3

from pylibcugraph._cugraph_c.random cimport (
    cugraph_rng_state_t,
)

cdef class CuGraphRandomState:
    cdef cugraph_rng_state_t* rng_state_ptr
