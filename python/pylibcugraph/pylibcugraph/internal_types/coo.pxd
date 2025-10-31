# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Have cython use python 3 syntax
# cython: language_level = 3


from pylibcugraph._cugraph_c.coo cimport (
    cugraph_coo_t,
)
from pylibcugraph._cugraph_c.array cimport (
    cugraph_type_erased_device_array_view_t,
)

cdef class COO:
    cdef cugraph_coo_t* c_coo_ptr
    cdef set_ptr(self, cugraph_coo_t* ptr)
    cdef get_array(self, cugraph_type_erased_device_array_view_t* ptr)
