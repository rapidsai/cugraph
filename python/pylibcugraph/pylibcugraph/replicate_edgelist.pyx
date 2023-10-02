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


from pylibcugraph._cugraph_c.resource_handle cimport (
    cugraph_resource_handle_t,
    bool_t,
)
from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)
from pylibcugraph._cugraph_c.array cimport (
    cugraph_type_erased_device_array_view_t,
    cugraph_type_erased_device_array_view_free,
)
from pylibcugraph._cugraph_c.graph_functions cimport (
    cugraph_allgather_edgelist,
    cugraph_induced_subgraph_result_t,
    cugraph_induced_subgraph_get_sources,
    cugraph_induced_subgraph_get_destinations,
    cugraph_induced_subgraph_get_edge_weights,
    cugraph_induced_subgraph_get_subgraph_offsets,
    cugraph_induced_subgraph_result_free,
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)
from pylibcugraph.resource_handle cimport (
    ResourceHandle,
)
from pylibcugraph.graphs cimport (
    _GPUGraph,
)
from pylibcugraph.utils cimport (
    assert_success,
    assert_CAI_type,
    copy_to_cupy_array,
    create_cugraph_type_erased_device_array_view_from_py_obj
)


def replicate_edgelist(ResourceHandle resource_handle,
                       src_array,
                       dst_array,
                       weight_array):
    """
        Compute vertex pairs that are two hops apart. The resulting pairs are
        sorted before returning.

        Parameters
        ----------
        resource_handle : ResourceHandle
            Handle to the underlying device resources needed for referencing data
            and running algorithms.
        
        src_array : device array type
            Device array containing the vertex identifiers of the source of each
            directed edge. The order of the array corresponds to the ordering of the
            dst_array, where the ith item in src_array and the ith item in dst_array
            define the ith edge of the graph.
        
        dst_array : device array type
            Device array containing the vertex identifiers of the destination of
            each directed edge. The order of the array corresponds to the ordering
            of the src_array, where the ith item in src_array and the ith item in
            dst_array define the ith edge of the graph.

        weight_array : device array type
            Device array containing the weight values of each directed edge. The
            order of the array corresponds to the ordering of the src_array and
            dst_array arrays, where the ith item in weight_array is the weight value
            of the ith edge of the graph.

        Returns
        -------
        return a cupy arrays of 'first' and 'second' or a 'cugraph_vertex_pairs_t'
        which can be directly passed to the similarity algorithm?
    """
    #print("src array before all gather= ", src_array.to_cupy())
    #print("dst array before all gather= ", dst_array.to_cupy())
    #print("wgt array before all gather= ", weight_array.to_cupy())
    assert_CAI_type(src_array, "src_array")
    assert_CAI_type(dst_array, "dst_array")
    assert_CAI_type(weight_array, "weight_array", True)
    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr

    cdef cugraph_induced_subgraph_result_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    cdef cugraph_type_erased_device_array_view_t* start_vertices_ptr

    
    cdef cugraph_type_erased_device_array_view_t* srcs_view_ptr = \
        create_cugraph_type_erased_device_array_view_from_py_obj(src_array)
        
    cdef cugraph_type_erased_device_array_view_t* dsts_view_ptr = \
        create_cugraph_type_erased_device_array_view_from_py_obj(dst_array)

    
    cdef cugraph_type_erased_device_array_view_t* weights_view_ptr = \
        create_cugraph_type_erased_device_array_view_from_py_obj(weight_array)



    error_code = cugraph_allgather_edgelist(c_resource_handle_ptr,
                                            srcs_view_ptr,
                                            dsts_view_ptr,
                                            weights_view_ptr,
                                            &result_ptr,
                                            &error_ptr)
    assert_success(error_code, error_ptr, "replicate_edgelist")

    
    # Extract individual device array pointers from result and copy to cupy
    # arrays for returning.
    cdef cugraph_type_erased_device_array_view_t* sources_ptr = \
        cugraph_induced_subgraph_get_sources(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* destinations_ptr = \
        cugraph_induced_subgraph_get_destinations(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* edge_weights_ptr = \
        cugraph_induced_subgraph_get_edge_weights(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* subgraph_offsets_ptr = \
        cugraph_induced_subgraph_get_subgraph_offsets(result_ptr)

    # FIXME: Get ownership of the result data instead of performing a copy
    # for perfomance improvement
    cupy_sources = copy_to_cupy_array(
        c_resource_handle_ptr, sources_ptr)
    cupy_destinations = copy_to_cupy_array(
        c_resource_handle_ptr, destinations_ptr)
    cupy_edge_weights = copy_to_cupy_array(
        c_resource_handle_ptr, edge_weights_ptr)
    
    # FIXME: Check if there are weights before copying.
    cupy_subgraph_offsets = copy_to_cupy_array(
        c_resource_handle_ptr, subgraph_offsets_ptr)

    # Free pointer
    cugraph_induced_subgraph_result_free(result_ptr)

    #print("Done doing the replication")
    #print("src in plc = ", cupy_sources)
    #print("dst in plc = ", cupy_destinations)
    #print("weights in plc = ", cupy_edge_weights)

    return (cupy_sources, cupy_destinations,
                cupy_edge_weights, cupy_subgraph_offsets)
