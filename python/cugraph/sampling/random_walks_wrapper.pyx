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

from cugraph.sampling.random_walks cimport call_random_walks
from cugraph.structure.graph_primtypes cimport *
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from cugraph.structure import graph_primtypes_wrapper
import cudf
import rmm
import numpy as np
import numpy.ctypeslib as ctypeslib
from rmm._lib.device_buffer cimport DeviceBuffer
from cudf.core.buffer import Buffer
from cython.operator cimport dereference as deref

def random_walks(input_graph, start_vertices, max_depth):
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

    cdef uintptr_t c_src_vertices = src.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_dst_vertices = dst.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_edge_weights = <uintptr_t>NULL
    if weights is not None:
        c_edge_weights = weights.__cuda_array_interface__['data'][0]
        weight_t = weights.dtype
    else:
        weight_t = np.dtype("float32")

    # Pointers for random_walks
    cdef uintptr_t c_start_vertex_ptr = start_vertices.__cuda_array_interface__['data'][0]
    num_paths = start_vertices.size

    cdef unique_ptr[handle_t] handle_ptr
    handle_ptr.reset(new handle_t())
    handle_ = handle_ptr.get();

    cdef graph_container_t graph_container
    populate_graph_container(graph_container,
                             handle_[0],
                             <void*>c_src_vertices, <void*>c_dst_vertices, <void*>c_edge_weights,
                             <void*>NULL,
                             <numberTypeEnum>(<int>(numberTypeMap[vertex_t])),
                             <numberTypeEnum>(<int>(numberTypeMap[edge_t])),
                             <numberTypeEnum>(<int>(numberTypeMap[weight_t])),
                             num_partition_edges,
                             num_verts,
                             num_edges,
                             False,
                             False, False)

    if(vertex_t == np.dtype("int32")):
        if(edge_t == np.dtype("int32")):
            rw_ret_ptr = move(call_random_walks[int, int]( deref(handle_),
                                                           graph_container,
                                                           <int*> c_start_vertex_ptr,
                                                           <int> num_paths,
                                                           <int> max_depth))
        else: # (edge_t == np.dtype("int64")):
            rw_ret_ptr = move(call_random_walks[int, long]( deref(handle_),
                                                           graph_container,
                                                           <int*> c_start_vertex_ptr,
                                                           <long> num_paths,
                                                           <long> max_depth))
    else: # (vertex_t == edge_t == np.dtype("int64")):
        rw_ret_ptr = move(call_random_walks[long, long]( deref(handle_),
                                                           graph_container,
                                                           <long*> c_start_vertex_ptr,
                                                           <long> num_paths,
                                                           <long> max_depth))
