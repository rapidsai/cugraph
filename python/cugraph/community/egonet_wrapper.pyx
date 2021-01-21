# Copyright (c) 2021, NVIDIA CORPORATION.
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

def egonet(G, vertices, radius=1):
    """
    Call egonet
    """
    # Step 1: Declare the different varibales
    cdef graph_container_t graph_container
    # FIXME: Offsets and indices are currently hardcoded to int, but this may
    #        not be acceptable in the future.
    numberTypeMap = {np.dtype("int32") : <int>numberTypeEnum.int32Type,
                     np.dtype("int64") : <int>numberTypeEnum.int64Type,
                     np.dtype("float32") : <int>numberTypeEnum.floatType,
                     np.dtype("double") : <int>numberTypeEnum.doubleType}

    # Pointers required for CSR Graph
    cdef uintptr_t c_offsets_ptr        = <uintptr_t> NULL # Pointer to the CSR offsets
    cdef uintptr_t c_indices_ptr        = <uintptr_t> NULL # Pointer to the CSR indices
    cdef uintptr_t c_weights = <uintptr_t>NULL
    cdef uintptr_t c_local_verts = <uintptr_t> NULL;
    cdef uintptr_t c_local_edges = <uintptr_t> NULL;
    cdef uintptr_t c_local_offsets = <uintptr_t> NULL;
    weight_t = np.dtype("float32")

    # Pointers for egonet
    cdef uintptr_t c_source_vertex_ptr     = <uintptr_t> NULL # Pointer to the DataFrame 'vertex' Series

    # Step 2: Verifiy input_graph has the expected format 
    #TODO is this how it should be done with non-legacy ? Notice we are not passing CSR later
    if input_graph.adjlist is None:
        input_graph.view_adj_list()

    cdef unique_ptr[handle_t] handle_ptr
    handle_ptr.reset(new handle_t())
    handle_ = handle_ptr.get();

    # Step 3: Extract CSR offsets, indices, weights are not expected
    #         - offsets: int (signed, 32-bit)
    #         - indices: int (signed, 32-bit)
    #TODO what about 64b types? the backend now supports this 
    #TODO is this how it should be done with non-legacy? Can't find populate_graph_container for that
    [offsets, indices] = graph_primtypes_wrapper.datatype_cast([input_graph.adjlist.offsets, input_graph.adjlist.indices], [np.int32])
    c_offsets_ptr = offsets.__cuda_array_interface__['data'][0]
    c_indices_ptr = indices.__cuda_array_interface__['data'][0]

    # Step 4: Setup number of vertices and edges
    num_verts = input_graph.number_of_vertices()
    num_edges = input_graph.number_of_edges(directed_edges=True)

    # Step 5: Check if source index is valid
    if not 0 <= start < num_verts:
        raise ValueError("Starting vertex should be between 0 to number of vertices")

    # Step 6: Generate the result
    
    # Step 7: Associate <uintptr_t> to cudf Series for offsets
    #TODO check this
    c_source_vertex_ptr = vertices.__cuda_array_interface__['data'][0]
    n_subgraphs = vertices.size

    # Step 8: Proceed to egonet
    #TODO this should probaby accept csr no? maybe add another populate_graph_container for this case?
    cdef graph_container_t graph_container
    populate_graph_container(graph_container,
                             handle_[0],
                             <void*>c_src_vertices, <void*>c_dst_vertices, <void*>c_edge_weights,
                             <void*>c_vertex_partition_offsets,
                             <numberTypeEnum>(<int>(numberTypeMap[vertex_t])),
                             <numberTypeEnum>(<int>(numberTypeMap[edge_t])),
                             <numberTypeEnum>(<int>(numberTypeMap[weight_t])),
                             num_verts,
                             num_edges, num_edges,
                             False,
                             False, False) 

    el_struct = c_egonet.call_egonet[int, float](handle_ptr.get()[0],
                               graph_container,
                               <int*> c_source_vertex_ptr,
                               <int> n_subgraphs,
                               <int> radius)
    #TODO get the strcut and populate the offset serie and a graph


    return tmp
