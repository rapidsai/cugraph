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

from pylibcugraph._cugraph_c.lookup_src_dst cimport (
    cugraph_lookup_result_t,
    cugraph_lookup_result_free,
    cugraph_lookup_result_get_dsts,
    cugraph_lookup_result_get_srcs,
)
from pylibcugraph._cugraph_c.array cimport (
    cugraph_type_erased_device_array_view_t,
)
from pylibcugraph.utils cimport (
    create_cupy_array_view_for_device_ptr,
)

cdef class EdgeIdLookupResult:
    def __cinit__(self):
        """
        Sets this object as the owner of the given pointer.
        """
        self.result_c_ptr = NULL

    cdef set_ptr(self, cugraph_lookup_result_t* ptr):
        self.result_c_ptr = ptr

    def __dealloc__(self):
        if self.result_c_ptr is not NULL:
            cugraph_lookup_result_free(self.result_c_ptr)

    cdef get_array(self, cugraph_type_erased_device_array_view_t* ptr):
        if ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(
            ptr,
            self,
        )

    def get_sources(self):
        if self.result_c_ptr is NULL:
            return None
        cdef cugraph_type_erased_device_array_view_t* ptr = cugraph_lookup_result_get_srcs(self.result_c_ptr)
        return self.get_array(ptr)

    def get_destinations(self):
        if self.result_c_ptr is NULL:
            return None
        cdef cugraph_type_erased_device_array_view_t* ptr = cugraph_lookup_result_get_dsts(self.result_c_ptr)
        return self.get_array(ptr)
