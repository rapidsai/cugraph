# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

cdef extern from "cugraph_c/coo.h":
    ctypedef struct cugraph_coo_t:
        pass

    ctypedef struct cugraph_coo_list_t:
        pass

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_coo_get_sources(
            cugraph_coo_t* coo
        )

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_coo_get_destinations(
            cugraph_coo_t* coo
        )

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_coo_get_edge_weights(
            cugraph_coo_t* coo
        )

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_coo_get_edge_id(
            cugraph_coo_t* coo
        )

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_coo_get_edge_type(
            cugraph_coo_t* coo
        )

    cdef size_t \
        cugraph_coo_list_size(
            const cugraph_coo_list_t* coo_list
        )

    cdef cugraph_coo_t* \
        cugraph_coo_list_element(
            cugraph_coo_list_t* coo_list,
            size_t index)

    cdef void \
        cugraph_coo_free(
            cugraph_coo_t* coo
        )

    cdef void \
        cugraph_coo_list_free(
            cugraph_coo_list_t* coo_list
        )
