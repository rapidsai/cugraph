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

from cugraph.structure.graph_new cimport *
from cugraph.structure.graph_new cimport get_two_hop_neighbors as c_get_two_hop_neighbors
from cugraph.structure.utils_wrapper import *
from libcpp cimport bool
from libc.stdint cimport uintptr_t

import cudf
import rmm
import numpy as np


def datatype_cast(cols, dtypes):
    cols_out = []
    for col in cols:
        if col is None or col.dtype.type in dtypes:
            cols_out.append(col)
        else:
            cols_out.append(col.astype(dtypes[0]))
    return cols_out


def view_adj_list(input_graph):

    if input_graph.adjlist is None:
        if input_graph.edgelist is None:
            raise Exception('Graph is Empty')

        [src, dst] = datatype_cast([input_graph.edgelist.edgelist_df['src'], input_graph.edgelist.edgelist_df['dst']], [np.int32])
        weights = None
        if input_graph.edgelist.weights:
            [weights] = datatype_cast([input_graph.edgelist.edgelist_df['weights']], [np.float32, np.float64])

        return coo2csr(src, dst, weights)


def view_transposed_adj_list(input_graph):

    if input_graph.transposedadjlist is None:
        if input_graph.edgelist is None:
            if input_graph.adjlist is None:
                raise Exception('Graph is Empty')
            else:
                input_graph.view_edge_list()

        [src, dst] = datatype_cast([input_graph.edgelist.edgelist_df['src'], input_graph.edgelist.edgelist_df['dst']], [np.int32])
        weights = None
        if input_graph.edgelist.weights:
            [weights] = datatype_cast([input_graph.edgelist.edgelist_df['weights']], [np.float32, np.float64])

        return coo2csr(dst, src, weights)


def view_edge_list(input_graph):

    if input_graph.adjlist is None:
        raise Exception('Graph is Empty')

    [offsets, indices] = datatype_cast([input_graph.adjlist.offsets, input_graph.adjlist.indices], [np.int32])
    [weights] = datatype_cast([input_graph.adjlist.weights], [np.float32, np.float64])
    num_verts = input_graph.number_of_vertices()
    num_edges = len(indices)

    cdef uintptr_t c_offsets = offsets.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_indices = indices.__cuda_array_interface__['data'][0]
    cdef GraphCSRView[int,int,float] graph
    graph = GraphCSRView[int,int,float](<int*>c_offsets, <int*>c_indices, <float*>NULL, num_verts, num_edges)

    src_indices = cudf.Series(np.zeros(num_edges), dtype= indices.dtype)
    cdef uintptr_t c_src_indices = src_indices.__cuda_array_interface__['data'][0]
    graph.get_source_indices(<int*>c_src_indices)

    return src_indices, indices, weights


def _degree_coo(src, dst, x=0):
    #
    #  Computing the degree of the input graph from COO
    #
    cdef DegreeDirection dir

    if x == 0:
        dir = DIRECTION_IN_PLUS_OUT
    elif x == 1:
        dir = DIRECTION_IN
    elif x == 2:
        dir = DIRECTION_OUT
    else:
        raise Exception("x should be 0, 1 or 2")

    [src, dst] = datatype_cast([src, dst], [np.int32])

    num_verts = 1 + max(src.max(), dst.max())
    num_edges = len(src)

    vertex_col = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    degree_col = cudf.Series(np.zeros(num_verts, dtype=np.int32))

    cdef GraphCOOView[int,int,float] graph

    cdef uintptr_t c_vertex = vertex_col.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_degree = degree_col.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_src = src.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_dst = dst.__cuda_array_interface__['data'][0]

    graph = GraphCOOView[int,int,float](<int*>c_src, <int*>c_dst, <float*>NULL, num_verts, num_edges)

    graph.degree(<int*> c_degree, dir)
    graph.get_vertex_identifiers(<int*>c_vertex)

    return vertex_col, degree_col


def _degree_csr(offsets, indices, x=0):
    cdef DegreeDirection dir

    if x == 0:
        dir = DIRECTION_IN_PLUS_OUT
    elif x == 1:
        dir = DIRECTION_IN
    elif x == 2:
        dir = DIRECTION_OUT
    else:
        raise Exception("x should be 0, 1 or 2")

    [offsets, indices] = datatype_cast([offsets, indices], [np.int32])

    num_verts = len(offsets)-1
    num_edges = len(indices)

    vertex_col = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    degree_col = cudf.Series(np.zeros(num_verts, dtype=np.int32))

    cdef GraphCSRView[int,int,float] graph

    cdef uintptr_t c_vertex = vertex_col.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_degree = degree_col.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_offsets = offsets.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_indices = indices.__cuda_array_interface__['data'][0]

    graph = GraphCSRView[int,int,float](<int*>c_offsets, <int*>c_indices, <float*>NULL, num_verts, num_edges)
        
    graph.degree(<int*> c_degree, dir)
    graph.get_vertex_identifiers(<int*>c_vertex)

    return vertex_col, degree_col


def _degree(input_graph, x=0):
    transpose_x = { 0: 0,
                    2: 1,
                    1: 2 }
    
    if input_graph.adjlist is not None:
        return _degree_csr(input_graph.adjlist.offsets,
                           input_graph.adjlist.indices,
                           x)

    if input_graph.transposedadjlist is not None:
        return _degree_csr(input_graph.transposedadjlist.offsets,
                           input_graph.transposedadjlist.indices,
                           transpose_x[x])

    if input_graph.edgelist is not None:
        return _degree_coo(input_graph.edgelist.edgelist_df['src'],
                           input_graph.edgelist.edgelist_df['dst'],
                           x)
                           
    raise Exception("input_graph not COO, CSR or CSC")

    
def _degrees(input_graph):
    verts, indegrees = _degree(input_graph,1)
    verts, outdegrees = _degree(input_graph, 2)
    
    return verts, indegrees, outdegrees


def get_two_hop_neighbors(input_graph):
    cdef GraphCSRView[int,int,float] graph

    offsets = None
    indices = None
    transposed = False

    if input_graph.adjlist:
        [offsets, indices] = datatype_cast([input_graph.adjlist.offsets, input_graph.adjlist.indices], [np.int32])
    elif input_graph.transposedadjlist:
        [offsets, indices] = datatype_cast([input_graph.adjlist.offsets, input_graph.adjlist.indices], [np.int32])
        transposed = True
    else:
        input_graph.view_adj_list()
        [offsets, indices] = datatype_cast([input_graph.adjlist.offsets, input_graph.adjlist.indices], [np.int32])

    cdef uintptr_t c_offsets = offsets.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_indices = indices.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_first = <uintptr_t> NULL
    cdef uintptr_t c_second = <uintptr_t> NULL

    num_verts = input_graph.number_of_vertices()
    num_edges = len(indices)

    graph = GraphCSRView[int,int,float](<int*>c_offsets, <int*> c_indices, <float*>NULL, num_verts, num_edges)

    count = c_get_two_hop_neighbors(graph, <int**> &c_first, <int**> &c_second)
    
    df = cudf.DataFrame()
    df['first'] = rmm.device_array_from_ptr(c_first,
                                            nelem=count,
                                            dtype=np.int32)
    df['second'] = rmm.device_array_from_ptr(c_second,
                                             nelem=count,
                                             dtype=np.int32)

    return df


def weight_type(input_graph):
    weights_type = None
    if input_graph.edgelist.weights:
        weights_type = input_graph.edgelist.edgelist_df['weights'].dtype
    return weights_type
