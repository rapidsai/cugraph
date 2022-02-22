# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

from libc.stdint cimport uintptr_t
from libcpp.memory cimport unique_ptr

import cudf
import numpy as np

from cugraph.structure import graph_primtypes_wrapper
from cugraph.structure.graph_utilities cimport (graph_container_t,
                                                numberTypeEnum,
                                                populate_graph_container,
                                               )
from raft.common.handle cimport handle_t
from cugraph.link_analysis cimport hits as c_hits


def hits(input_graph, max_iter=100, tol=1.0e-5, nstart=None, normalized=True):
    """
    Call HITS, return a DataFrame containing the hubs and authorities for each
    vertex.
    """
    cdef graph_container_t graph_container

    numberTypeMap = {np.dtype("int32") : <int>numberTypeEnum.int32Type,
                     np.dtype("int64") : <int>numberTypeEnum.int64Type,
                     np.dtype("float32") : <int>numberTypeEnum.floatType,
                     np.dtype("double") : <int>numberTypeEnum.doubleType}


    if nstart is not None:
        raise ValueError('nstart is not currently supported')

    # Inputs
    vertex_t = np.dtype("int32")
    edge_t = np.dtype("int32")
    weight_t = np.dtype("float32")

    [src, dst] = graph_primtypes_wrapper.datatype_cast(
        [input_graph.edgelist.edgelist_df['src'],
         input_graph.edgelist.edgelist_df['dst']],
        [np.int32])
    weights = None
    cdef uintptr_t c_src_vertices = src.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_dst_vertices = dst.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_edge_weights = <uintptr_t>NULL

    num_verts = input_graph.number_of_vertices()
    num_edges = input_graph.number_of_edges(directed_edges=True)
    is_symmetric = not input_graph.is_directed()

    cdef unique_ptr[handle_t] handle_ptr
    handle_ptr.reset(new handle_t())
    handle_ = handle_ptr.get();

    populate_graph_container(graph_container,
                             handle_[0],
                             <void*>c_src_vertices, <void*>c_dst_vertices, <void*>c_edge_weights,
                             <void*>NULL,
                             <void*>NULL,
                             0,
                             <numberTypeEnum>(<int>(numberTypeMap[vertex_t])),
                             <numberTypeEnum>(<int>(numberTypeMap[edge_t])),
                             <numberTypeEnum>(<int>(numberTypeMap[weight_t])),
                             num_edges,
                             num_verts, num_edges,
                             False,
                             is_symmetric,
                             False,
                             False)

    # Outputs
    df = cudf.DataFrame()
    df['hubs'] = cudf.Series(np.zeros(num_verts, dtype=np.float32))
    df['authorities'] = cudf.Series(np.zeros(num_verts, dtype=np.float32))
    # The vertex Series is simply the renumbered vertex IDs, which is just 0 to (num_verts-1)
    df['vertex'] = cudf.Series(np.arange(num_verts, dtype=np.int32))

    cdef uintptr_t c_hubs_ptr = df['hubs'].__cuda_array_interface__['data'][0];
    cdef uintptr_t c_authorities_ptr = df['authorities'].__cuda_array_interface__['data'][0];

    # Call HITS
    c_hits.call_hits[int, float](handle_ptr.get()[0],
                                 graph_container,
                                 <float*> c_hubs_ptr,
                                 <float*> c_authorities_ptr,
                                 max_iter,
                                 tol,
                                 <float*> NULL,
                                 normalized)

    return df
