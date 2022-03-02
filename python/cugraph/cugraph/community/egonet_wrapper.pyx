# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

import numpy as np
from libc.stdint cimport uintptr_t
from libcpp cimport bool
from libcpp.utility cimport move
from libcpp.memory cimport unique_ptr

import cudf

from cugraph.structure.graph_utilities cimport (populate_graph_container,
                                                graph_container_t,
                                                numberTypeEnum,
                                                move as graph_utils_move,
                                                )
from cugraph.structure.graph_primtypes cimport move_device_buffer_to_column
from raft.common.handle cimport handle_t
from cugraph.structure import graph_primtypes_wrapper
from cugraph.community.egonet cimport call_egonet
from raft.common.handle cimport handle_t
from raft.common.handle import Handle


def egonet(input_graph, vertices, radius=1):
    """
    Call egonet
    """
    # FIXME: Offsets and indices are currently hardcoded to int, but this may
    #        not be acceptable in the future.
    numberTypeMap = {np.dtype("int32") : <int>numberTypeEnum.int32Type,
                     np.dtype("int64") : <int>numberTypeEnum.int64Type,
                     np.dtype("float32") : <int>numberTypeEnum.floatType,
                     np.dtype("double") : <int>numberTypeEnum.doubleType}

    [src, dst] = graph_primtypes_wrapper.datatype_cast([input_graph.edgelist.edgelist_df['src'], input_graph.edgelist.edgelist_df['dst']], [np.int32])
    vertex_t = src.dtype
    edge_t = np.dtype("int32")
    weights = None
    if input_graph.edgelist.weights:
        weights = input_graph.edgelist.edgelist_df['weights']

    num_verts = input_graph.number_of_vertices()
    num_edges = input_graph.number_of_edges(directed_edges=True)
    num_local_edges = num_edges

    cdef uintptr_t c_src_vertices = src.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_dst_vertices = dst.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_edge_weights = <uintptr_t>NULL
    if weights is not None:
        c_edge_weights = weights.__cuda_array_interface__['data'][0]
        weight_t = weights.dtype
        is_weighted = True
    else:
        weight_t = np.dtype("float32")
        is_weighted = False

    is_symmetric = not input_graph.is_directed()

    # Pointers for egonet
    vertices = vertices.astype('int32')
    cdef uintptr_t c_source_vertex_ptr = vertices.__cuda_array_interface__['data'][0]
    n_subgraphs = vertices.size
    n_streams = 1
    if n_subgraphs > 1 :
        n_streams = min(n_subgraphs, 32)
    cdef unique_ptr[handle_t] handle_ptr
    handle = Handle(n_streams=n_streams)
    cdef handle_t* handle_ = <handle_t*><size_t> handle.getHandle()

    cdef graph_container_t graph_container
    populate_graph_container(graph_container,
                             handle_[0],
                             <void*>c_src_vertices, <void*>c_dst_vertices, <void*>c_edge_weights,
                             <void*>NULL,
                             <void*>NULL,
                             0,
                             <numberTypeEnum>(<int>(numberTypeMap[vertex_t])),
                             <numberTypeEnum>(<int>(numberTypeMap[edge_t])),
                             <numberTypeEnum>(<int>(numberTypeMap[weight_t])),
                             num_local_edges,
                             num_verts,
                             num_edges,
                             is_weighted,
                             is_symmetric,
                             False, False)

    if(weight_t==np.dtype("float32")):
        el_struct_ptr = graph_utils_move(
            call_egonet[int, float](handle_[0],
                                    graph_container,
                                    <int*> c_source_vertex_ptr,
                                    <int> n_subgraphs,
                                    <int> radius))
    else:
        el_struct_ptr = graph_utils_move(
            call_egonet[int, double](handle_[0],
                                     graph_container,
                                     <int*> c_source_vertex_ptr,
                                     <int> n_subgraphs,
                                     <int> radius))

    el_struct = move(el_struct_ptr.get()[0])
    src = move_device_buffer_to_column(move(el_struct.src_indices), vertex_t)
    dst = move_device_buffer_to_column(move(el_struct.dst_indices), vertex_t)
    wgt = move_device_buffer_to_column(move(el_struct.edge_data), weight_t)

    df = cudf.DataFrame()
    df['src'] = src
    df['dst'] = dst
    if wgt is not None:
        df['weight'] = wgt

    offsets = move_device_buffer_to_column(move(el_struct.subgraph_offsets), "int")

    return df, offsets
