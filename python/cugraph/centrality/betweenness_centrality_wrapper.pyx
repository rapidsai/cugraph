# Copyright (c) 2019, NVIDIA CORPORATION.
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

from cugraph.centrality.betweenness_centrality cimport betweenness_centrality as c_betweenness_centrality
from cugraph.structure.graph_new cimport *
from cugraph.utilities.column_utils cimport *
from cugraph.utilities.unrenumber import unrenumber
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
from cugraph.structure import graph_wrapper
import cudf
import cudf._lib as libcudf
import rmm
import numpy as np
import numpy.ctypeslib as ctypeslib


def betweenness_centrality(input_graph):
    """
    Call betweenness centrality
    """

    if not input_graph.adjlist:
        input_graph.view_adj_list()

    num_verts = input_graph.number_of_vertices()
    num_edges = input_graph.number_of_edges()

    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    df['betweenness_centrality'] = cudf.Series(np.zeros(num_verts, dtype=np.float32))

    cdef uintptr_t c_identifier = df['vertex'].__cuda_array_interface__['data'][0];
    cdef uintptr_t c_betweenness = df['betweenness_centrality'].__cuda_array_interface__['data'][0];

    cdef uintptr_t c_offsets = input_graph.adjlist.offsets.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_indices = input_graph.adjlist.indices.__cuda_array_interface__['data'][0]

    cdef GraphCSR[int,int,float] graph
    
    graph = GraphCSR[int,int,float](<int*>c_offsets, <int*>c_indices, <float*>NULL, num_verts, num_edges)

    c_betweenness_centrality[int,int,float,float](graph, <float*> c_betweenness)
    graph.get_vertex_identifiers(<int*>c_identifier)

    if input_graph.renumbered:
        df = unrenumber(input_graph.edgelist.renumber_map, df, 'vertex')

    return df
