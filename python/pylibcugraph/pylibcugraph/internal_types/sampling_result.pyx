# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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


from pylibcugraph._cugraph_c.array cimport (
    cugraph_type_erased_device_array_view_t,
)
from pylibcugraph._cugraph_c.algorithms cimport (
    cugraph_sample_result_t,
    cugraph_sample_result_get_sources,
    cugraph_sample_result_get_destinations,
    cugraph_sample_result_get_edge_weight,
    cugraph_sample_result_get_edge_id,
    cugraph_sample_result_get_edge_type,
    cugraph_sample_result_get_hop,
    cugraph_sample_result_get_start_labels,
    cugraph_sample_result_free,
)
from pylibcugraph.utils cimport (
    create_cupy_array_view_for_device_ptr,
)


cdef class SamplingResult:
    """
    Cython interface to a cugraph_sample_result_t pointer. Instances of this
    call will take ownership of the pointer and free it under standard python
    GC rules (ie. when all references to it are no longer present).

    This class provides methods to return non-owning cupy ndarrays for the
    corresponding array members. Returning these cupy arrays increments the ref
    count on the SamplingResult instances from which the cupy arrays are
    referencing.
    """
    def __cinit__(self):
        # This SamplingResult instance owns sample_result_ptr now. It will be
        # freed when this instance is deleted (see __dealloc__())
        self.c_sample_result_ptr = NULL

    def __dealloc__(self):
        if self.c_sample_result_ptr is not NULL:
            cugraph_sample_result_free(self.c_sample_result_ptr)

    cdef set_ptr(self, cugraph_sample_result_t* sample_result_ptr):
        self.c_sample_result_ptr = sample_result_ptr

    def get_sources(self):
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef cugraph_type_erased_device_array_view_t* device_array_view_ptr = (
            cugraph_sample_result_get_sources(self.c_sample_result_ptr)
        )
        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)

    def get_destinations(self):
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef cugraph_type_erased_device_array_view_t* device_array_view_ptr = (
            cugraph_sample_result_get_destinations(self.c_sample_result_ptr)
        )
        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)

    def get_edge_weights(self):
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef cugraph_type_erased_device_array_view_t* device_array_view_ptr = (
            cugraph_sample_result_get_edge_weight(self.c_sample_result_ptr)
        )
        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)

    def get_indices(self):
        return self.get_edge_weights()
    
    def get_edge_ids(self):
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef cugraph_type_erased_device_array_view_t* device_array_view_ptr = (
            cugraph_sample_result_get_edge_id(self.c_sample_result_ptr)
        )
        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)

    def get_edge_types(self):
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef cugraph_type_erased_device_array_view_t* device_array_view_ptr = (
            cugraph_sample_result_get_edge_type(self.c_sample_result_ptr)
        )
        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)
    
    def get_batch_ids(self):
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef cugraph_type_erased_device_array_view_t* device_array_view_ptr = (
            cugraph_sample_result_get_start_labels(self.c_sample_result_ptr)
        )
        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)
                                
    def get_hop_ids(self):
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef cugraph_type_erased_device_array_view_t* device_array_view_ptr = (
            cugraph_sample_result_get_hop(self.c_sample_result_ptr)
        )
        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)