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

from pylibcugraph._cugraph_c.resource_handle cimport (
    cugraph_resource_handle_t,
)
from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)
from pylibcugraph._cugraph_c.array cimport (
    cugraph_type_erased_device_array_view_t,
    cugraph_type_erased_device_array_view_create,
    cugraph_type_erased_device_array_view_free,
    cugraph_type_erased_host_array_view_t,
    cugraph_type_erased_host_array_view_create,
    cugraph_type_erased_host_array_view_free,
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)
from pylibcugraph._cugraph_c.lookup_src_dst cimport (
    cugraph_lookup_container_t,
    cugraph_build_edge_id_and_type_to_src_dst_lookup_map,
    cugraph_lookup_container_free,
    cugraph_lookup_endpoints_from_edge_ids_and_single_type,
    cugraph_lookup_result_t,
)
from pylibcugraph.utils cimport (
    assert_success,
    assert_CAI_type,
    assert_AI_type,
    get_c_type_from_numpy_type,
    create_cugraph_type_erased_device_array_view_from_py_obj
)
from pylibcugraph.resource_handle cimport (
    ResourceHandle,
)
from pylibcugraph.graphs cimport (
    _GPUGraph,
)
from pylibcugraph.internal_types.edge_id_lookup_result cimport (
    EdgeIdLookupResult,
)

cdef class EdgeIdLookupTable:
    def __cinit__(self, ResourceHandle resource_handle, _GPUGraph graph):
        self.handle = resource_handle
        self.graph = graph

        cdef cugraph_error_code_t error_code
        cdef cugraph_error_t* error_ptr

        error_code = cugraph_build_edge_id_and_type_to_src_dst_lookup_map(
            <cugraph_resource_handle_t*>self.handle.c_resource_handle_ptr,
            <cugraph_graph_t*>self.graph.c_graph_ptr,
            &self.lookup_container_c_ptr,
            &error_ptr,
        )

        assert_success(error_code, error_ptr, "cugraph_build_edge_id_and_type_to_src_dst_lookup_map")

    def __dealloc__(self):
        if self.lookup_container_c_ptr is not NULL:
            cugraph_lookup_container_free(self.lookup_container_c_ptr)

    def lookup_vertex_ids(
        self,
        edge_ids,
        int edge_type
    ):
        """
        For a single edge type, finds the source and destination vertex ids corresponding
        to the provided edge ids.
        """

        cdef cugraph_error_code_t error_code
        cdef cugraph_error_t* error_ptr
        cdef cugraph_lookup_result_t* result_ptr

        cdef cugraph_type_erased_device_array_view_t* edge_ids_c_ptr
        edge_ids_c_ptr = create_cugraph_type_erased_device_array_view_from_py_obj(edge_ids)

        error_code = cugraph_lookup_endpoints_from_edge_ids_and_single_type(
            <cugraph_resource_handle_t*>self.handle.c_resource_handle_ptr,
            <cugraph_graph_t*>self.graph.c_graph_ptr,
            self.lookup_container_c_ptr,
            edge_ids_c_ptr,
            edge_type,
            &result_ptr,
            &error_ptr,
        )

        assert_success(error_code, error_ptr, "cugraph_lookup_endpoints_from_edge_ids_and_single_type")

        lr = EdgeIdLookupResult()
        lr.set_ptr(<cugraph_lookup_result_t*>(result_ptr))
        return {
            'sources': lr.get_sources(),
            'destinations': lr.get_destinations(),
        }
