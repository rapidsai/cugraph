# Copyright (c) 2020, NVIDIA CORPORATION.
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

from cugraph.link_analysis.hits cimport hits as c_hits
from cugraph.structure.graph_new cimport *
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from cugraph.structure import graph_new_wrapper
import cudf
import rmm
import numpy as np
import numpy.ctypeslib as ctypeslib


def hits(input_graph, max_iter=100, tol=1.0e-5, nstart=None, normalized=True):
    """
    Call HITS
    """

    if nstart is not None:
        raise ValueError('nstart is not currently supported')

    if not input_graph.adjlist:
        input_graph.view_adj_list()

    [offsets, indices] = graph_new_wrapper.datatype_cast([input_graph.adjlist.offsets, input_graph.adjlist.indices], [np.int32])

    num_verts = input_graph.number_of_vertices()
    num_edges = input_graph.number_of_edges(directed_edges=True)

    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    df['hubs'] = cudf.Series(np.zeros(num_verts, dtype=np.float32))
    df['authorities'] = cudf.Series(np.zeros(num_verts, dtype=np.float32))

    #cdef bool normalized = <bool> 1

    cdef uintptr_t c_identifier = df['vertex'].__cuda_array_interface__['data'][0];
    cdef uintptr_t c_hubs = df['hubs'].__cuda_array_interface__['data'][0];
    cdef uintptr_t c_authorities = df['authorities'].__cuda_array_interface__['data'][0];

    cdef uintptr_t c_offsets = offsets.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_indices = indices.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights = <uintptr_t>NULL

    cdef GraphCSRView[int,int,float] graph_float
    
    graph_float = GraphCSRView[int,int,float](<int*>c_offsets, <int*>c_indices, <float*>c_weights, num_verts, num_edges)

    c_hits[int,int,float](graph_float, max_iter, tol, <float*> NULL,
                          normalized, <float*>c_hubs, <float*>c_authorities);
    graph_float.get_vertex_identifiers(<int*>c_identifier)

    if input_graph.renumbered:
        # FIXME: multi-column vertex support
        tmp = input_graph.edgelist.renumber_map.from_vertex_id(df['vertex'])
        df['vertex'] = tmp['0']

    return df
