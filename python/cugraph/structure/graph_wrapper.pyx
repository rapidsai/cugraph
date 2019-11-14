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

from cugraph.structure.c_graph cimport *
from cugraph.utilities.column_utils cimport *
from cudf._lib.cudf cimport np_dtype_from_gdf_column
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

import cudf
import cudf._lib as libcudf
import rmm
import numpy as np


def allocate_cpp_graph():
    cdef gdf_graph * g
    g = <gdf_graph*> calloc(1, sizeof(gdf_graph))

    cdef uintptr_t graph_ptr = <uintptr_t> g

    return graph_ptr

def release_cpp_graph(graph_ptr):
    cdef uintptr_t graph = graph_ptr
    cdef gdf_graph * g = <gdf_graph*> graph
    free(g)

def renumber(source_col, dest_col):
    cdef gdf_column src_renumbered
    cdef gdf_column dst_renumbered
    cdef gdf_column numbering_map

    cdef gdf_column source = get_gdf_column_view(source_col)
    cdef gdf_column dest = get_gdf_column_view(dest_col)

    err = gdf_renumber_vertices(&source,
                                &dest,
                                &src_renumbered,
                                &dst_renumbered,
                                &numbering_map)

    libcudf.cudf.check_gdf_error(err)

    src_renumbered_array = rmm.device_array_from_ptr(<uintptr_t> src_renumbered.data,
                                 nelem=src_renumbered.size,
                                 dtype=np_dtype_from_gdf_column(&src_renumbered))
    dst_renumbered_array = rmm.device_array_from_ptr(<uintptr_t> dst_renumbered.data,
                                 nelem=dst_renumbered.size,
                                 dtype=np_dtype_from_gdf_column(&dst_renumbered))
    numbering_map_array = rmm.device_array_from_ptr(<uintptr_t> numbering_map.data,
                                 nelem=numbering_map.size,
                                 dtype=np_dtype_from_gdf_column(&numbering_map))

    return cudf.Series(src_renumbered_array), cudf.Series(dst_renumbered_array), cudf.Series(numbering_map_array)

def add_edge_list(graph_ptr, source_col, dest_col, value_col=None):
    cdef uintptr_t graph = graph_ptr
    cdef gdf_graph * g = <gdf_graph*> graph

    # Checks in python
    if len(source_col) != len(dest_col):
        raise ValueError("Source and Destination length mismatch")
    if source_col.dtype.type is not np.int32:
        source_col = source_col.astype(np.int32)
    if dest_col.dtype.type is not np.int32:
        dest_col = dest_col.astype(np.int32)
    if value_col is not None and value_col.dtype.type is not np.float32 and value_col.dtype.type is not np.float64:
        value_col = value_col.astype(np.float32) 

    cdef gdf_column c_source_col = get_gdf_column_view(source_col)
    cdef gdf_column c_dest_col = get_gdf_column_view(dest_col)
    cdef gdf_column c_value_col
    cdef gdf_column * c_value_col_ptr
    if value_col is None:
        c_value_col_ptr = NULL
    else:
        c_value_col = get_gdf_column_view(value_col)
        c_value_col_ptr = &c_value_col

    err = gdf_edge_list_view(g,
                             &c_source_col,
                             &c_dest_col,
                             c_value_col_ptr)
    libcudf.cudf.check_gdf_error(err)

def get_edge_list(graph_ptr):
    cdef uintptr_t graph = graph_ptr
    cdef gdf_graph * g = <gdf_graph*> graph

    # we should add get_number_of_edges() to gdf_graph (and this should be
    # used instead of g.edgeList.src_indices.size)
    col_size = g.edgeList.src_indices.size

    cdef uintptr_t src_col_data = <uintptr_t> g.edgeList.src_indices.data
    cdef uintptr_t dest_col_data = <uintptr_t> g.edgeList.dest_indices.data
    cdef uintptr_t value_col_data = <uintptr_t> NULL
    if g.edgeList.edge_data is not NULL:
        value_col_data = <uintptr_t> g.edgeList.edge_data.data

    # g.edgeList.src_indices.data, g.edgeList.dest_indices.data, and
    # g.edgeList.edge_data.data are not owned by this instance, so should not
    # be freed when the resulting cudf.Series objects are finalized (this will
    # lead to double free, and undefined behavior). The finalizer parameter of
    # rmm.device_array_from_ptr shuold be ``None`` (default value) for this
    # purpose (instead of rmm._finalizer(handle, stream)).

    src_data = rmm.device_array_from_ptr(
                   src_col_data,
                   nelem=col_size,
                   dtype=np.int32)
    source_col = cudf.Series(src_data)

    dest_data = rmm.device_array_from_ptr(
                    dest_col_data,
                    nelem=col_size,
                    dtype=np.int32)
    dest_col = cudf.Series(dest_data)

    value_col = None
    if <void*>value_col_data is not NULL:
        value_data = rmm.device_array_from_ptr(
                         value_col_data,
                         nelem=col_size,
                         dtype=np_dtype_from_gdf_column(g.edgeList.edge_data))
        value_col = cudf.Series(value_data)

    return source_col, dest_col, value_col

def add_adj_list(graph_ptr, offset_col, index_col, value_col=None):
    cdef uintptr_t graph = graph_ptr
    cdef gdf_graph * g = <gdf_graph*> graph
    
    if offset_col.dtype.type is not np.int32:
        offset_col = offset_col.astype(np.int32)
    if index_col.dtype.type is not np.int32:
        index_col = index_col.astype(np.int32)
    if value_col is not None and value_col.dtype.type is not np.float32 and value_col.dtype.type is not np.float64:
        value_col = value_col.astype(np.float32)

    cdef gdf_column c_offset_col = get_gdf_column_view(offset_col)
    cdef gdf_column c_index_col = get_gdf_column_view(index_col)
    cdef gdf_column c_value_col
    cdef gdf_column * c_value_col_ptr
    if value_col is None:
        c_value_col_ptr = NULL
    else:
        c_value_col = get_gdf_column_view(value_col)
        c_value_col_ptr = &c_value_col

    err = gdf_adj_list_view(g,
                            &c_offset_col,
                            &c_index_col,
                            c_value_col_ptr)
    libcudf.cudf.check_gdf_error(err)

def get_adj_list(graph_ptr):
    cdef uintptr_t graph = graph_ptr
    cdef gdf_graph * g = <gdf_graph*> graph

    offset_col_size = g.adjList.offsets.size
    index_col_size = g.adjList.indices.size

    cdef uintptr_t offset_col_data = <uintptr_t> g.adjList.offsets.data
    cdef uintptr_t index_col_data = <uintptr_t> g.adjList.indices.data
    cdef uintptr_t value_col_data = <uintptr_t> NULL
    if g.adjList.edge_data is not NULL:
        value_col_data = <uintptr_t> g.adjList.edge_data.data

    # g.adjList.offsets.data, g.adjList.indices.data, and
    # g.adjList.edge_data.data are not owned by this instance, so should not be
    # freed here (this will lead to double free, and undefined behavior). The
    # finalizer parameter of rmm.device_array_from_ptr shuold be ``None``
    # (default value) for this purpose (instead of
    # rmm._finalizer(handle, stream)).

    offset_data = rmm.device_array_from_ptr(
                       offset_col_data,
                       nelem=offset_col_size,
                       dtype=np.int32)
    offset_col = cudf.Series(offset_data)

    index_data = rmm.device_array_from_ptr(
                       index_col_data,
                       nelem=index_col_size,
                       dtype=np.int32)
    index_col = cudf.Series(index_data)

    value_col = None
    if <void*>value_col_data is not NULL:
        value_data = rmm.device_array_from_ptr(
                         value_col_data,
                         nelem=index_col_size,
                         dtype=np_dtype_from_gdf_column(g.adjList.edge_data))
        value_col = cudf.Series(value_data)

    return offset_col, index_col, value_col

def view_edge_list(input_graph):
    cdef uintptr_t graph = allocate_cpp_graph()
    cdef gdf_graph * g = <gdf_graph*> graph
    if input_graph.edgelist is None:
        if input_graph.adjlist is None:
            raise Exception('Graph is Empty')
        else:
            add_adj_list(graph, input_graph.adjlist.offsets, input_graph.adjlist.indices, input_graph.adjlist.weights)
            err = gdf_add_edge_list(g)
            libcudf.cudf.check_gdf_error(err)
            source, dest, value = get_edge_list(graph)
            input_graph.edgelist = input_graph.EdgeList(source, dest, value)

def view_adj_list(input_graph):
    cdef uintptr_t graph = allocate_cpp_graph()
    cdef gdf_graph * g = <gdf_graph*> graph
    if input_graph.adjlist is None:
        if input_graph.edgelist is None:
            raise Exception('Graph is Empty')
        else:
            if len(input_graph.edgelist.edgelist_df.columns)>2:
                add_edge_list(graph, input_graph.edgelist.edgelist_df['src'], input_graph.edgelist.edgelist_df['dst'], input_graph.edgelist.edgelist_df['weights'])
            else:
                add_edge_list(graph, input_graph.edgelist.edgelist_df['src'], input_graph.edgelist.edgelist_df['dst'])
            err = gdf_add_adj_list(g)
            libcudf.cudf.check_gdf_error(err)
            offsets, indices, values = get_adj_list(graph)
            input_graph.adjlist = input_graph.AdjList(offsets, indices, values)

def add_transposed_adj_list(graph_ptr, offset_col, index_col, value_col=None):
    cdef uintptr_t graph = graph_ptr
    cdef gdf_graph * g = <gdf_graph*> graph

    if offset_col.dtype.type is not np.int32:
        offset_col = offset_col.astype(np.int32)
    if index_col.dtype.type is not np.int32:
        index_col = index_col.astype(np.int32)
    if value_col is not None and value_col.dtype.type is not np.float32 and value_col.dtype.type is not np.float64:
        value_col = value_col.astype(np.float32)

    cdef gdf_column c_offset_col = get_gdf_column_view(offset_col)
    cdef gdf_column c_index_col = get_gdf_column_view(index_col)
    cdef gdf_column c_value_col
    cdef gdf_column * c_value_col_ptr
    if value_col is None:
        c_value_col_ptr = NULL
    else:
        c_value_col = get_gdf_column_view(value_col)
        c_value_col_ptr = &c_value_col

    err = gdf_transposed_adj_list_view(g,
                            &c_offset_col,
                            &c_index_col,
                            c_value_col_ptr)
    libcudf.cudf.check_gdf_error(err)

def get_transposed_adj_list(graph_ptr):
    cdef uintptr_t graph = graph_ptr
    cdef gdf_graph * g = <gdf_graph*> graph

    offset_col_size = g.transposedAdjList.offsets.size
    index_col_size = g.transposedAdjList.indices.size

    cdef uintptr_t offset_col_data = <uintptr_t> g.transposedAdjList.offsets.data
    cdef uintptr_t index_col_data = <uintptr_t> g.transposedAdjList.indices.data
    cdef uintptr_t value_col_data = <uintptr_t> NULL
    if g.transposedAdjList.edge_data is not NULL:
        value_col_data = <uintptr_t> g.transposedAdjList.edge_data.data

    # g.transposedAdjList.offsets.data, g.transposedAdjList.indices.data and
    # g.transposedAdjList.edge_data.data are not owned by this instance, so
    # should not be freed here (this will lead to double free, and undefined
    # behavior). The finalizer parameter of rmm.device_array_from_ptr should
    # be ``None`` (default value) for this purpose (instead of
    # rmm._finalizer(handle, stream)).

    offset_data = rmm.device_array_from_ptr(
                       offset_col_data,
                       nelem=offset_col_size,
                       dtype=np.int32)
    offset_col = cudf.Series(offset_data)

    index_data = rmm.device_array_from_ptr(
                     index_col_data,
                     nelem=index_col_size,
                     dtype=np.int32)
    index_col = cudf.Series(index_data)

    value_col = None
    if <void*>value_col_data is not NULL:
        value_data = rmm.device_array_from_ptr(
                         value_col_data,
                         nelem=index_col_size,
                         dtype=np_dtype_from_gdf_column(g.transposedAdjList.edge_data))
        value_col = cudf.Series(value_data)

    return offset_col, index_col, value_col

def get_two_hop_neighbors(input_graph):
    cdef uintptr_t graph = allocate_cpp_graph()
    cdef gdf_graph * g = <gdf_graph*> graph

    if input_graph.adjlist:
        add_adj_list(graph, input_graph.adjlist.offsets, input_graph.adjlist.indices, input_graph.adjlist.weights)
    else:
        if len(input_graph.edgelist.edgelist_df.columns)>2:
            add_edge_list(graph, input_graph.edgelist.edgelist_df['src'], input_graph.edgelist.edgelist_df['dst'], input_graph.edgelist.edgelist_df['weights'])
        else:
            add_edge_list(graph, input_graph.edgelist.edgelist_df['src'], input_graph.edgelist.edgelist_df['dst'])
        err = gdf_add_adj_list(g)
        libcudf.cudf.check_gdf_error(err)
        offsets, indices, values = get_adj_list(graph)
        input_graph.adjlist = input_graph.AdjList(offsets, indices, values)

    cdef gdf_column c_first_col
    cdef gdf_column c_second_col
    err = gdf_get_two_hop_neighbors(g, &c_first_col, &c_second_col)
    libcudf.cudf.check_gdf_error(err)
    df = cudf.DataFrame()
    if c_first_col.dtype == GDF_INT32:
        first_out = rmm.device_array_from_ptr(<uintptr_t>c_first_col.data,
                                              nelem=c_first_col.size,
                                              dtype=np.int32)
        second_out = rmm.device_array_from_ptr(<uintptr_t>c_second_col.data,
                                               nelem=c_second_col.size,
                                               dtype=np.int32)
        df['first'] = first_out
        df['second'] = second_out
    if c_first_col.dtype == GDF_INT64:
        first_out = rmm.device_array_from_ptr(<uintptr_t>c_first_col.data,
                                              nelem=c_first_col.size,
                                              dtype=np.int64)
        second_out = rmm.device_array_from_ptr(<uintptr_t>c_second_col.data,
                                               nelem=c_second_col.size,
                                               dtype=np.int64)
        df['first'] = first_out
        df['second'] = second_out

    return df

def number_of_vertices(input_graph):
    cdef uintptr_t graph = allocate_cpp_graph()
    cdef gdf_graph * g = <gdf_graph*> graph

    if input_graph.adjlist:
        add_adj_list(graph, input_graph.adjlist.offsets, input_graph.adjlist.indices, input_graph.adjlist.weights)
    else:
        if input_graph.edgelist.weights:
            add_edge_list(graph, input_graph.edgelist.edgelist_df['src'], input_graph.edgelist.edgelist_df['dst'], input_graph.edgelist.edgelist_df['weights'])
        else:
            add_edge_list(graph, input_graph.edgelist.edgelist_df['src'], input_graph.edgelist.edgelist_df['dst'])
        err = gdf_number_of_vertices(g)
        libcudf.cudf.check_gdf_error(err)
    return g.numberOfVertices


def number_of_edges(graph_ptr):
    cdef uintptr_t graph = graph_ptr
    cdef gdf_graph * g = <gdf_graph*> graph
    if g.adjList:
        return g.adjList.indices.size
    elif g.transposedAdjList:
        return g.transposedAdjList.indices.size
    elif g.edgeList:
        return g.edgeList.src_indices.size
    else:
        # An empty graph
        return 0

def _degree(input_graph, x=0):
    cdef uintptr_t graph = allocate_cpp_graph()
    cdef gdf_graph * g = <gdf_graph*> graph

    if input_graph.adjlist:
        add_adj_list(graph, input_graph.adjlist.offsets, input_graph.adjlist.indices, input_graph.adjlist.weights)
    else:
        if len(input_graph.edgelist.edgelist_df.columns)>2:
            add_edge_list(graph, input_graph.edgelist.edgelist_df['src'], input_graph.edgelist.edgelist_df['dst'], input_graph.edgelist.edgelist_df['weights'])
        else:
            add_edge_list(graph, input_graph.edgelist.edgelist_df['src'], input_graph.edgelist.edgelist_df['dst'])
        err = gdf_add_adj_list(g)
        libcudf.cudf.check_gdf_error(err)
        offsets, indices, values = get_adj_list(graph)
        input_graph.adjlist = input_graph.AdjList(offsets, indices, values)

    n = number_of_vertices(input_graph)

    vertex_col = cudf.Series(np.zeros(n, dtype=np.int32))
    c_vertex_col = get_gdf_column_view(vertex_col)
    err = g.adjList.get_vertex_identifiers(&c_vertex_col)
    libcudf.cudf.check_gdf_error(err)

    degree_col = cudf.Series(np.zeros(n, dtype=np.int32))
    cdef gdf_column c_degree_col = get_gdf_column_view(degree_col)
    err = gdf_degree(g, &c_degree_col, <int>x)
    libcudf.cudf.check_gdf_error(err)

    return vertex_col, degree_col

def _degrees(input_graph):
    cdef uintptr_t graph = allocate_cpp_graph()
    cdef gdf_graph * g = <gdf_graph*> graph

    if input_graph.adjlist:
        add_adj_list(graph, input_graph.adjlist.offsets, input_graph.adjlist.indices, input_graph.adjlist.weights)
    else:
        if len(input_graph.edgelist.edgelist_df.columns)>2:
            add_edge_list(graph, input_graph.edgelist.edgelist_df['src'], input_graph.edgelist.edgelist_df['dst'], input_graph.edgelist.edgelist_df['weights'])
        else:
            add_edge_list(graph, input_graph.edgelist.edgelist_df['src'], input_graph.edgelist.edgelist_df['dst'])
        err = gdf_add_adj_list(g)
        libcudf.cudf.check_gdf_error(err)
        offsets, indices, values = get_adj_list(graph)
        input_graph.adjlist = input_graph.AdjList(offsets, indices, values)
    
    n = number_of_vertices(input_graph)

    vertex_col = cudf.Series(np.zeros(n, dtype=np.int32))
    c_vertex_col = get_gdf_column_view(vertex_col)
    err = g.adjList.get_vertex_identifiers(&c_vertex_col)
    libcudf.cudf.check_gdf_error(err)

    in_degree_col = cudf.Series(np.zeros(n, dtype=np.int32))
    cdef gdf_column c_in_degree_col = get_gdf_column_view(in_degree_col)
    err = gdf_degree(g, &c_in_degree_col, <int>1)
    libcudf.cudf.check_gdf_error(err)

    out_degree_col = cudf.Series(np.zeros(n, dtype=np.int32))
    cdef gdf_column c_out_degree_col = get_gdf_column_view(out_degree_col)
    err = gdf_degree(g, &c_out_degree_col, <int>2)
    libcudf.cudf.check_gdf_error(err)

    return vertex_col, in_degree_col, out_degree_col
