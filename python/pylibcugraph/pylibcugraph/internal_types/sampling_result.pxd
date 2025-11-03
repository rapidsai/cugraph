# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Have cython use python 3 syntax
# cython: language_level = 3


from pylibcugraph._cugraph_c.algorithms cimport (
    cugraph_sample_result_t,
)


cdef class SamplingResult:
    cdef cugraph_sample_result_t* c_sample_result_ptr
    cdef set_ptr(self, cugraph_sample_result_t* sample_result_ptr)
