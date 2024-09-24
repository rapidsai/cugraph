# Copyright (c) 2024, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Have cython use python 3 syntax
# cython: language_level = 3

from pylibcugraph._cugraph_c.coo cimport (
    cugraph_coo_t,
    cugraph_coo_free,
    cugraph_coo_get_sources,
    cugraph_coo_get_destinations,
    cugraph_coo_get_edge_weights,
    cugraph_coo_get_edge_id,
    cugraph_coo_get_edge_type,
)
from pylibcugraph._cugraph_c.array cimport (
    cugraph_type_erased_device_array_view_t,
)
from pylibcugraph.utils cimport create_cupy_array_view_for_device_ptr

cdef class COO:
    """
    Cython interface to a cugraph_coo_t pointer. Instances of this
    call will take ownership of the pointer and free it under standard python
    GC rules (ie. when all references to it are no longer present).

    This class provides methods to return non-owning cupy ndarrays for the
    corresponding array members. Returning these cupy arrays increments the ref
    count on the COO instances from which the cupy arrays are
    referencing.
    """
    def __cinit__(self):
        # This COO instance owns sample_result_ptr now. It will be
        # freed when this instance is deleted (see __dealloc__())
        self.c_coo_ptr = NULL

    def __dealloc__(self):
        if self.c_coo_ptr is not NULL:
            cugraph_coo_free(self.c_coo_ptr)

    cdef set_ptr(self, cugraph_coo_t* ptr):
        self.c_coo_ptr = ptr

    cdef get_array(self, cugraph_type_erased_device_array_view_t* ptr):
        if ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(
            ptr,
            self,
        )

    def get_sources(self):
        if self.c_coo_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef cugraph_type_erased_device_array_view_t* ptr = cugraph_coo_get_sources(self.c_coo_ptr)
        return self.get_array(ptr)

    def get_destinations(self):
        if self.c_coo_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef cugraph_type_erased_device_array_view_t* ptr = cugraph_coo_get_destinations(self.c_coo_ptr)
        return self.get_array(ptr)

    def get_edge_ids(self):
        if self.c_coo_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef cugraph_type_erased_device_array_view_t* ptr = cugraph_coo_get_edge_id(self.c_coo_ptr)
        return self.get_array(ptr)

    def get_edge_types(self):
        if self.c_coo_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef cugraph_type_erased_device_array_view_t* ptr = cugraph_coo_get_edge_type(self.c_coo_ptr)
        return self.get_array(ptr)

    def get_edge_weights(self):
        if self.c_coo_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef cugraph_type_erased_device_array_view_t* ptr = cugraph_coo_get_edge_weights(self.c_coo_ptr)
        return self.get_array(ptr)
