# Copyright (c) 2022, NVIDIA CORPORATION.
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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

cimport cugraph.sampling.node2vec as c_node2vec
from cugraph.structure.graph_utilities cimport *
from cugraph.structure import graph_primtypes_wrapper
from libcpp cimport bool
from libc.stdint cimport uintptr_t
import cudf
import numpy as np


def node2vec(input_graph, sources, max_depth, p, q):
    """
    Call node2vec
    """

    """
    # Step 1: Declare the different variables
    cdef graph_container_t graph_container

    numberTypeMap = {np.dtype("int32") : <int>numberTypeEnum.int32Type,
                     np.dtype("int64") : <int>numberTypeEnum.int64Type,
                     np.dtype("float32") : <int>numberTypeEnum.floatType,
                     np.dtype("double") : <int>numberTypeEnum.doubleType}

    # Pointers required for CSR Graph
    cdef uintptr_t c_offsets_ptr        = <uintptr_t> NULL # Pointer to the CSR offsets
    cdef uintptr_t c_indices_ptr        = <uintptr_t> NULL # Pointer to the CSR indices
    cdef uintptr_t c_weights_ptr        = <uintptr_t> NULL # Pointer to the CSR weights
    cdef uintptr_t c_local_verts = <uintptr_t> NULL;
    cdef uintptr_t c_local_edges = <uintptr_t> NULL;
    cdef uintptr_t c_local_offsets = <uintptr_t> NULL;
    weight_t = np.dtype("float32")

    # Pointers for node2vec
    cdef uintptr_t c_paths_ptr          = <uintptr_t> NULL # Pointer to the DataFrame paths
    cdef uintptr_t c_weights2_ptr       = <uintptr_t> NULL # Pointer to the DataFrame weights
    cdef uintptr_t c_offsets2_ptr       = <uintptr_t> NULL # Pointer to the DataFrame offsets

    cdef uintptr_t c_sources_ptr        = <uintptr_t> NULL # Pointer to the DataFrame offsets

    cdef unique_ptr[handle_t] handle_ptr
    handle_ptr.reset(new handle_t())
    handle_ = handle_ptr.get();

    # Step 2: Verify that input_graph has the expected format
    if not input_graph.adjlist:
        input_graph.view_adj_list()

    # Step 3: Extract CSR offsets, indices and weights
    #         - offsets: int (signed, 32-bit)
    #         - indices: int (signed, 32-bit)
    #         - weights: float / double
    #         Extract data_type from weights (not None: float / double, None: signed int 32-bit)
    [offsets, indices] = graph_primtypes_wrapper.datatype_cast([input_graph.adjlist.offsets, input_graph.adjlist.indices], [np.int32])
    [weights] = graph_primtypes_wrapper.datatype_cast([input_graph.adjlist.weights], [np.float32, np.float64])
    c_offsets_ptr = offsets.__cuda_array_interface__['data'][0]
    c_indices_ptr = indices.__cuda_array_interface__['data'][0]

    if weights is not None:
        weight_t = weights.dtype
        c_weights_ptr = weights.__cuda_array_interface__['data'][0]

    # Step 4: Setup number of vertices and number of edges
    num_verts = input_graph.number_of_vertices()
    num_edges = input_graph.number_of_edges(directed_edges=True)

    # Step 5: Check if source indices are valid
    for source in sources:
        if not 0 <= source < num_verts:
            raise ValueError("Starting vertex should be between 0 to number of vertices")

    # Step 6: Generation of the result cudf.DataFrame
    #         Distances depends on data_type (c.f. Step 3)
    df = cudf.DataFrame()

    df['paths'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    df['weights'] = cudf.Series(np.zeros(num_verts, dtype=weight_t))
    df['offsets'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))

    # Step 7: Associate <uintptr_t> to cudf Series
    c_paths_ptr = df['paths'].__cuda_array_interface__['data'][0]
    c_weights2_ptr = df['weights'].__cuda_array_interface__['data'][0]
    c_offsets2_ptr = df['offsets'].__cuda_array_interface__['data'][0]

    c_sources_ptr = sources.__cuda_array_interface__['data'][0]

    # Step 8: Call node2vec, note this is not correct
    populate_graph_container_legacy(graph_container,
                                    <graphTypeEnum>(<int>(graphTypeEnum.LegacyCSR)),
                                    handle_[0],
                                    <void*>c_offsets_ptr, <void*>c_indices_ptr, <void*>c_weights_ptr,
                                    <numberTypeEnum>(<int>(numberTypeEnum.int32Type)),
                                    <numberTypeEnum>(<int>(numberTypeEnum.int32Type)),
                                    <numberTypeEnum>(<int>(numberTypeMap[weight_t])),
                                    num_verts, num_edges,
                                    <int*>c_local_verts, <int*>c_local_edges, <int*>c_local_offsets)

    if weight_t == np.float32:
        c_node2vec.call_node2vec[int, float](handle_[0],
                               graph_container,
                               <int*> c_paths_ptr,
                               <float*> c_weights2_ptr,
                               <int*> c_offsets2_ptr,
                               <int*> c_sources_ptr,
                               <float> p,
                               <float> q)
    elif weight_t == np.float64:
        c_node2vec.call_node2vec[int, double](handle_[0],
                               graph_container,
                               <int*> c_paths_ptr,
                               <double*> c_weights2_ptr,
                               <int*> c_offsets2_ptr,
                               <int*> c_sources_ptr,
                               <float> p,
                               <float> q)

    # Stubbed out code - layer 2 being the wrapper in cugraph
    return df
    """

    # FIXME: To remove when implemented
    return 222
