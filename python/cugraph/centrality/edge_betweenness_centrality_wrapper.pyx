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

from cugraph.centrality.betweenness_centrality cimport edge_betweenness_centrality as c_edge_betweenness_centrality
from cugraph.structure import graph_new_wrapper
from cugraph.structure.graph import DiGraph, Graph
from cugraph.structure.graph_new cimport *
from cugraph.utilities.unrenumber import unrenumber
from libc.stdint cimport uintptr_t
from libcpp cimport bool
import cudf
import numpy as np
import numpy.ctypeslib as ctypeslib


def get_output_df(input_graph, result_dtype):
    number_of_edges = input_graph.number_of_edges(directed_edges=True)
    df = cudf.DataFrame()
    df['src'] = cudf.Series(np.zeros(number_of_edges, dtype=np.int32))
    df['dst'] = input_graph.adjlist.indices.copy()
    df['betweenness_centrality'] = cudf.Series(np.zeros(number_of_edges,
                                               dtype=result_dtype))
    return df


def edge_betweenness_centrality(input_graph, normalized, weight, k,
                                vertices, result_dtype):
    """
    Call betweenness centrality
    """
    cdef GraphCSRViewFloat graph_float
    cdef GraphCSRViewDouble graph_double
    cdef uintptr_t c_src_identifier = <uintptr_t> NULL
    cdef uintptr_t c_dst_identifier = <uintptr_t> NULL
    cdef uintptr_t c_betweenness = <uintptr_t> NULL
    cdef uintptr_t c_vertices = <uintptr_t> NULL
    cdef uintptr_t c_weight = <uintptr_t> NULL

    if not input_graph.adjlist:
        input_graph.view_adj_list()

    df = get_output_df(input_graph, result_dtype)

    c_src_identifier = df['src'].__cuda_array_interface__['data'][0]
    c_dst_identifier = df['dst'].__cuda_array_interface__['data'][0]
    c_betweenness = df['betweenness_centrality'].__cuda_array_interface__['data'][0]

    if weight is not None:
        c_weight = weight.__cuda_array_interface__['data'][0]

    if vertices is not None:
        np_verts = np.array(vertices, dtype=np.int32)
        c_vertices = np_verts.__array_interface__['data'][0]

    c_k = 0
    if k is not None:
        c_k = k

    # NOTE: The current implementation only has <int, int, float, float> and
    #       <int, int, double, double> as explicit template declaration
    #       The current BFS requires the GraphCSR to be declared
    #       as <int, int, float> or <int, int double> even if weights is null
    if result_dtype == np.float32:
        graph_float = get_graph_view[GraphCSRViewFloat](input_graph, False)
        # fixme: there might be a way to avoid manually setting the graph property
        graph_float.prop.directed = type(input_graph) is DiGraph

        c_edge_betweenness_centrality[int, int,
                                      float, float](get_default_raft_handle(),
                                                    graph_float,
                                                    <float*> c_betweenness,
                                                    normalized,
                                                    <float*> c_weight,
                                                    c_k,
                                                    <int*> c_vertices)
        graph_float.get_source_indices(<int*>c_src_identifier)
    elif result_dtype == np.float64:
        graph_double = get_graph_view[GraphCSRViewDouble](input_graph, False)
        # FIXME: there might be a way to avoid manually setting
        #        the graph property
        graph_double.prop.directed = type(input_graph) is DiGraph

        c_edge_betweenness_centrality[int, int,
                                      double, double](get_default_raft_handle(),
                                                      graph_double,
                                                      <double*> c_betweenness,
                                                      normalized,
                                                      <double*> c_weight,
                                                      c_k,
                                                      <int*> c_vertices)
        graph_double.get_source_indices(<int*>c_src_identifier)
    else:
        raise TypeError("result type for betweenness centrality can only be "
                        "float or double")

    # Same as Betweenness Centrality unrenumber resuls might be organized
    # in buckets
    if input_graph.renumbered:
        df = unrenumber(input_graph.edgelist.renumber_map, df, 'src')
        df = unrenumber(input_graph.edgelist.renumber_map, df, 'dst')

    if type(input_graph) is Graph:
        lower_triangle = df['src'] >= df['dst']
        df[["src", "dst"]][lower_triangle] = df[["dst", "src"]][lower_triangle]
        df = df.groupby(by=["src", "dst"]).sum().reset_index()

    return df
