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
from cython.operator cimport dereference as deref

import cudf

from cugraph.structure.graph_utilities cimport (populate_graph_container,
                                                graph_container_t,
                                                numberTypeEnum,
                                                )
from raft.common.handle cimport handle_t
from cugraph.structure import graph_primtypes_wrapper
from cugraph.sampling.random_walks cimport (call_random_walks,
                                            call_rw_paths,
                                            random_walk_ret_t,
                                            random_walk_path_t,
                                            )
from cugraph.structure.graph_primtypes cimport (move_device_buffer_to_column,
                                                move_device_buffer_to_series,
                                                )

def random_walks(input_graph, start_vertices, max_depth, use_padding):
    """
    Call random_walks
    """
    # FIXME: Offsets and indices are currently hardcoded to int, but this may
    #        not be acceptable in the future.
    numberTypeMap = {np.dtype("int32") : <int>numberTypeEnum.int32Type,
                     np.dtype("int64") : <int>numberTypeEnum.int64Type,
                     np.dtype("float32") : <int>numberTypeEnum.floatType,
                     np.dtype("double") : <int>numberTypeEnum.doubleType}
    [src, dst] = [input_graph.edgelist.edgelist_df['src'], input_graph.edgelist.edgelist_df['dst']]
    vertex_t = src.dtype
    edge_t = np.dtype("int32")
    weights = None
    if input_graph.edgelist.weights:
        weights = input_graph.edgelist.edgelist_df['weights']
    num_verts = input_graph.number_of_vertices()
    num_edges = input_graph.number_of_edges(directed_edges=True)
    num_partition_edges = num_edges

    if num_edges > (2**31 - 1):
        edge_t = np.dtype("int64")
    cdef unique_ptr[random_walk_ret_t] rw_ret_ptr

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

    # Pointers for random_walks
    start_vertices = start_vertices.astype('int32')
    cdef uintptr_t c_start_vertex_ptr = start_vertices.__cuda_array_interface__['data'][0]
    num_paths = start_vertices.size
    cdef unique_ptr[handle_t] handle_ptr
    handle_ptr.reset(new handle_t())
    handle_ = handle_ptr.get()
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
                             num_partition_edges,
                             num_verts,
                             num_edges,
                             is_weighted,
                             is_symmetric,
                             False, False)
    if(vertex_t == np.dtype("int32")):
        if(edge_t == np.dtype("int32")):
            rw_ret_ptr = move(call_random_walks[int, int]( deref(handle_),
                                                           graph_container,
                                                           <int*> c_start_vertex_ptr,
                                                           <int> num_paths,
                                                           <int> max_depth,
                                                           <bool> use_padding))
        else: # (edge_t == np.dtype("int64")):
            rw_ret_ptr = move(call_random_walks[int, long]( deref(handle_),
                                                           graph_container,
                                                           <int*> c_start_vertex_ptr,
                                                           <long> num_paths,
                                                           <long> max_depth,
                                                           <bool> use_padding))
    else: # (vertex_t == edge_t == np.dtype("int64")):
        rw_ret_ptr = move(call_random_walks[long, long]( deref(handle_),
                                                           graph_container,
                                                           <long*> c_start_vertex_ptr,
                                                           <long> num_paths,
                                                           <long> max_depth,
                                                           <bool> use_padding))


    rw_ret = move(rw_ret_ptr.get()[0])
    set_vertex = move_device_buffer_to_series(
        move(rw_ret.d_coalesced_v_), vertex_t, "set_vertex")
    set_edge = move_device_buffer_to_series(
        move(rw_ret.d_coalesced_w_), weight_t, "set_edge")

    if not use_padding:
        set_sizes = move_device_buffer_to_series(
            move(rw_ret.d_sizes_), edge_t, "set_sizes")
    else:
        set_sizes = None

    return set_vertex, set_edge, set_sizes


def rw_path_retrieval(num_paths, sizes):
    cdef unique_ptr[handle_t] handle_ptr
    handle_ptr.reset(new handle_t())
    handle_ = handle_ptr.get()
    index_t = sizes.dtype

    cdef unique_ptr[random_walk_path_t] rw_path_ptr
    cdef uintptr_t c_sizes = sizes.__cuda_array_interface__['data'][0]

    if index_t == np.dtype("int32"):
        rw_path_ptr = move(call_rw_paths[int](deref(handle_),
                                              <int>num_paths,
                                              <int*>c_sizes))
    else: # index_t == np.dtype("int64"):
        rw_path_ptr = move(call_rw_paths[long](deref(handle_),
                                               <long>num_paths,
                                               <long*>c_sizes))

    rw_path = move(rw_path_ptr.get()[0])
    vertex_offsets = move_device_buffer_to_column(move(rw_path.d_v_offsets), index_t)
    weight_sizes = move_device_buffer_to_column(move(rw_path.d_w_sizes), index_t)
    weight_offsets = move_device_buffer_to_column(move(rw_path.d_w_offsets), index_t)

    df = cudf.DataFrame()
    df['vertex_offsets'] = vertex_offsets
    df['weight_sizes'] = weight_sizes
    df['weight_offsets'] = weight_offsets
    return df
