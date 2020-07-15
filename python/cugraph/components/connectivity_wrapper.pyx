# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

from cugraph.components.connectivity cimport *
from cugraph.structure.graph_new cimport *
from cugraph.structure import utils_wrapper
from cugraph.structure import graph_new_wrapper
from libc.stdint cimport uintptr_t
from cugraph.structure.symmetrize import symmetrize
from cugraph.structure.graph import Graph as type_Graph
from cugraph.utilities.unrenumber import unrenumber

import cudf
import numpy as np

def weakly_connected_components(input_graph):
    """
    Call connected_components
    """
    offsets = None
    indices = None
    
    if type(input_graph) is not type_Graph:
        #
        # Need to create a symmetrized CSR for this local
        # computation, don't want to keep it.
        #
        [src, dst] = graph_new_wrapper.datatype_cast([input_graph.edgelist.edgelist_df['src'],
                                                      input_graph.edgelist.edgelist_df['dst']],
                                                     [np.int32])
        src, dst = symmetrize(src, dst)
        [offsets, indices] = utils_wrapper.coo2csr(src, dst)[0:2]
    else:
        if not input_graph.adjlist:
            input_graph.view_adj_list()

        [offsets, indices] = graph_new_wrapper.datatype_cast([input_graph.adjlist.offsets,
                                                              input_graph.adjlist.indices],
                                                             [np.int32])

    num_verts = input_graph.number_of_vertices()
    num_edges = input_graph.number_of_edges(directed_edges=True)

    df = cudf.DataFrame()
    df['vertices'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    df['labels'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    
    cdef uintptr_t c_offsets    = offsets.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_indices    = indices.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_identifier = df['vertices'].__cuda_array_interface__['data'][0];
    cdef uintptr_t c_labels_val = df['labels'].__cuda_array_interface__['data'][0];

    cdef GraphCSRView[int,int,float] g

    g = GraphCSRView[int,int,float](<int*>c_offsets, <int*>c_indices, <float*>NULL, num_verts, num_edges)

    cdef cugraph_cc_t connect_type=CUGRAPH_WEAK
    connected_components(g, <cugraph_cc_t>connect_type, <int *>c_labels_val)

    g.get_vertex_identifiers(<int*>c_identifier)

    if input_graph.renumbered:
        df = unrenumber(input_graph.edgelist.renumber_map, df, 'vertices')

    return df


def strongly_connected_components(input_graph):
    """
    Call connected_components
    """
    if not input_graph.adjlist:
        input_graph.view_adj_list()

    [offsets, indices] = graph_new_wrapper.datatype_cast([input_graph.adjlist.offsets, input_graph.adjlist.indices], [np.int32])

    num_verts = input_graph.number_of_vertices()
    num_edges = input_graph.number_of_edges(directed_edges=True)

    df = cudf.DataFrame()
    df['vertices'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    df['labels'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    
    cdef uintptr_t c_offsets    = offsets.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_indices    = indices.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_identifier = df['vertices'].__cuda_array_interface__['data'][0];
    cdef uintptr_t c_labels_val = df['labels'].__cuda_array_interface__['data'][0];

    cdef GraphCSRView[int,int,float] g

    g = GraphCSRView[int,int,float](<int*>c_offsets, <int*>c_indices, <float*>NULL, num_verts, num_edges)

    cdef cugraph_cc_t connect_type=CUGRAPH_STRONG
    connected_components(g, <cugraph_cc_t>connect_type, <int *>c_labels_val)

    g.get_vertex_identifiers(<int*>c_identifier)

    if input_graph.renumbered:
        df = unrenumber(input_graph.edgelist.renumber_map, df, 'vertices')

    return df
