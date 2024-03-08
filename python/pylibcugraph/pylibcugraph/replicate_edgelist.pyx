# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
    cugraph_type_erased_device_array_view_free,
)
from pylibcugraph._cugraph_c.graph_functions cimport (
    cugraph_allgather,
    cugraph_induced_subgraph_result_t,
    cugraph_induced_subgraph_get_sources,
    cugraph_induced_subgraph_get_destinations,
    cugraph_induced_subgraph_get_edge_weights,
    cugraph_induced_subgraph_get_edge_ids,
    cugraph_induced_subgraph_get_edge_type_ids,
    cugraph_induced_subgraph_get_subgraph_offsets,
    cugraph_induced_subgraph_result_free,
)
from pylibcugraph.resource_handle cimport (
    ResourceHandle,
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
                       weight_array,
                       edge_id_array,
                       edge_type_id_array):
    """
        Replicate edges across all GPUs

        Parameters
        ----------
        resource_handle : ResourceHandle
            Handle to the underlying device resources needed for referencing data
            and running algorithms.

        src_array : device array type, optional
            Device array containing the vertex identifiers of the source of each
            directed edge. The order of the array corresponds to the ordering of the
            dst_array, where the ith item in src_array and the ith item in dst_array
            define the ith edge of the graph.

        dst_array : device array type, optional
            Device array containing the vertex identifiers of the destination of
            each directed edge. The order of the array corresponds to the ordering
            of the src_array, where the ith item in src_array and the ith item in
            dst_array define the ith edge of the graph.

        weight_array : device array type, optional
            Device array containing the weight values of each directed edge. The
            order of the array corresponds to the ordering of the src_array and
            dst_array arrays, where the ith item in weight_array is the weight value
            of the ith edge of the graph.

        edge_id_array : device array type, optional
            Device array containing the edge id values of each directed edge. The
            order of the array corresponds to the ordering of the src_array and
            dst_array arrays, where the ith item in edge_id_array is the id value
            of the ith edge of the graph.

        edge_type_id_array : device array type, optional
            Device array containing the edge type id values of each directed edge. The
            order of the array corresponds to the ordering of the src_array and
            dst_array arrays, where the ith item in edge_type_id_array is the type id
            value of the ith edge of the graph.

        Returns
        -------
        return cupy arrays of 'src' and/or 'dst' and/or 'weight'and/or 'edge_id'
        and/or 'edge_type_id'.
    """
    assert_CAI_type(src_array, "src_array", True)
    assert_CAI_type(dst_array, "dst_array", True)
    assert_CAI_type(weight_array, "weight_array", True)
    assert_CAI_type(edge_id_array, "edge_id_array", True)
    assert_CAI_type(edge_type_id_array, "edge_type_id_array", True)
    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr

    cdef cugraph_induced_subgraph_result_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    cdef cugraph_type_erased_device_array_view_t* srcs_view_ptr = \
        create_cugraph_type_erased_device_array_view_from_py_obj(src_array)

    cdef cugraph_type_erased_device_array_view_t* dsts_view_ptr = \
        create_cugraph_type_erased_device_array_view_from_py_obj(dst_array)


    cdef cugraph_type_erased_device_array_view_t* weights_view_ptr = \
        create_cugraph_type_erased_device_array_view_from_py_obj(weight_array)

    cdef cugraph_type_erased_device_array_view_t* edge_ids_view_ptr = \
        create_cugraph_type_erased_device_array_view_from_py_obj(edge_id_array)

    cdef cugraph_type_erased_device_array_view_t* edge_type_ids_view_ptr = \
        create_cugraph_type_erased_device_array_view_from_py_obj(edge_type_id_array)

    error_code = cugraph_allgather(c_resource_handle_ptr,
                                   srcs_view_ptr,
                                   dsts_view_ptr,
                                   weights_view_ptr,
                                   edge_ids_view_ptr,
                                   edge_type_ids_view_ptr,
                                   &result_ptr,
                                   &error_ptr)
    assert_success(error_code, error_ptr, "replicate_edgelist")
    # Extract individual device array pointers from result and copy to cupy
    # arrays for returning.
    cdef cugraph_type_erased_device_array_view_t* sources_ptr
    if src_array is not None:
        sources_ptr = cugraph_induced_subgraph_get_sources(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* destinations_ptr
    if dst_array is not None:
        destinations_ptr = cugraph_induced_subgraph_get_destinations(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* edge_weights_ptr = \
        cugraph_induced_subgraph_get_edge_weights(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* edge_ids_ptr = \
        cugraph_induced_subgraph_get_edge_ids(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* edge_type_ids_ptr = \
        cugraph_induced_subgraph_get_edge_type_ids(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* subgraph_offsets_ptr = \
        cugraph_induced_subgraph_get_subgraph_offsets(result_ptr)

    # FIXME: Get ownership of the result data instead of performing a copy
    # for perfomance improvement

    cupy_sources = None
    cupy_destinations = None
    cupy_edge_weights = None
    cupy_edge_ids = None
    cupy_edge_type_ids = None

    if src_array is not None:
        cupy_sources = copy_to_cupy_array(
            c_resource_handle_ptr, sources_ptr)

    if dst_array is not None:
        cupy_destinations = copy_to_cupy_array(
            c_resource_handle_ptr, destinations_ptr)

    if weight_array is not None:
        cupy_edge_weights = copy_to_cupy_array(
            c_resource_handle_ptr, edge_weights_ptr)

    if edge_id_array is not None:
        cupy_edge_ids = copy_to_cupy_array(
            c_resource_handle_ptr, edge_ids_ptr)

    if edge_type_id_array is not None:
        cupy_edge_type_ids = copy_to_cupy_array(
            c_resource_handle_ptr, edge_type_ids_ptr)

    cupy_subgraph_offsets = copy_to_cupy_array(
        c_resource_handle_ptr, subgraph_offsets_ptr)

    # Free pointer
    cugraph_induced_subgraph_result_free(result_ptr)
    if src_array is not None:
        cugraph_type_erased_device_array_view_free(srcs_view_ptr)
    if dst_array is not None:
        cugraph_type_erased_device_array_view_free(dsts_view_ptr)
    if weight_array is not None:
        cugraph_type_erased_device_array_view_free(weights_view_ptr)
    if edge_id_array is not None:
        cugraph_type_erased_device_array_view_free(edge_ids_view_ptr)
    if edge_type_id_array is not None:
        cugraph_type_erased_device_array_view_free(edge_type_ids_view_ptr)

    return (cupy_sources, cupy_destinations,
            cupy_edge_weights, cupy_edge_ids,
            cupy_edge_type_ids, cupy_subgraph_offsets)
