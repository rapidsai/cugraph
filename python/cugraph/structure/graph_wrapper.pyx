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
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

import cudf
from librmm_cffi import librmm as rmm
import numpy as np


dtypes_inv = {GDF_INT32: np.int32, GDF_INT64: np.int64, GDF_FLOAT32: np.float32, GDF_FLOAT64: np.float64}


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

    cudf.bindings.cudf_cpp.check_gdf_error(err)

    src_renumbered_array = rmm.device_array_from_ptr(<uintptr_t> src_renumbered.data,
                                 nelem=src_renumbered.size,
                                 dtype=dtypes_inv[src_renumbered.dtype])
    dst_renumbered_array = rmm.device_array_from_ptr(<uintptr_t> dst_renumbered.data,
                                 nelem=dst_renumbered.size,
                                 dtype=dtypes_inv[dst_renumbered.dtype])
    numbering_map_array = rmm.device_array_from_ptr(<uintptr_t> numbering_map.data,
                                 nelem=numbering_map.size,
                                 dtype=dtypes_inv[numbering_map.dtype])

    return cudf.Series(src_renumbered_array), cudf.Series(dst_renumbered_array), cudf.Series(numbering_map_array)

def add_edge_list(graph_ptr, source_col, dest_col, value_col=None):
    cdef uintptr_t graph = graph_ptr
    cdef gdf_graph * g = <gdf_graph*> graph

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
    cudf.bindings.cudf_cpp.check_gdf_error(err)

def view_edge_list(graph_ptr):
    cdef uintptr_t graph = graph_ptr
    cdef gdf_graph * g = <gdf_graph*> graph
    err = gdf_add_edge_list(g)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

    # we should add get_number_of_edges() to gdf_graph (and this should be
    # used instead of g.edgeList.src_indices.size)
    col_size = g.edgeList.src_indices.size

    cdef uintptr_t src_col_data = <uintptr_t> g.edgeList.src_indices.data
    cdef uintptr_t dest_col_data = <uintptr_t> g.edgeList.dest_indices.data

    src_data = rmm.device_array_from_ptr(src_col_data,
                                 nelem=col_size,
                                 dtype=np.int32)  # ,
                                 # finalizer=rmm._make_finalizer(src_col_data, 0))
    dest_data = rmm.device_array_from_ptr(dest_col_data,
                                 nelem=col_size,
                                 dtype=np.int32)  # ,
                                 # finalizer=rmm._make_finalizer(dest_col_data, 0))
    # g.edgeList.src_indices.data and g.edgeList.dest_indices.data are not
    # owned by this instance, so should not be freed here (this will lead
    # to double free, and undefined behavior).

    return cudf.Series(src_data), cudf.Series(dest_data)

def delete_edge_list(graph_ptr):
    cdef uintptr_t graph = graph_ptr
    err = gdf_delete_edge_list(<gdf_graph*> graph)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

def add_adj_list(graph_ptr, offset_col, index_col, value_col=None):
    cdef uintptr_t graph = graph_ptr
    cdef gdf_graph * g = <gdf_graph*> graph

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
    cudf.bindings.cudf_cpp.check_gdf_error(err)

def view_adj_list(graph_ptr):
    cdef uintptr_t graph = graph_ptr
    cdef gdf_graph * g = <gdf_graph*> graph
    err = gdf_add_adj_list(g)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

    offset_col_size = g.adjList.offsets.size
    index_col_size = g.adjList.indices.size

    cdef uintptr_t offset_col_data = <uintptr_t> g.adjList.offsets.data
    cdef uintptr_t index_col_data = <uintptr_t> g.adjList.indices.data

    offsets_data = rmm.device_array_from_ptr(offset_col_data,
                                 nelem=offset_col_size,
                                 dtype=np.int32) # ,
                                 # finalizer=rmm._make_finalizer(offset_col_data, 0))
    indices_data = rmm.device_array_from_ptr(index_col_data,
                                 nelem=index_col_size,
                                 dtype=np.int32) # ,
                                 # finalizer=rmm._make_finalizer(index_col_data, 0))
    # g.adjList.offsets.data and g.adjList.indices.data are not owned by
    # this instance, so should not be freed here (this will lead to double
    # free, and undefined behavior).

    return cudf.Series(offsets_data), cudf.Series(indices_data)

def delete_adj_list(graph_ptr):
    """
    Delete the adjacency list.
    """
    cdef uintptr_t graph = graph_ptr
    err = gdf_delete_adj_list(<gdf_graph*> graph)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

def add_transposed_adj_list(graph_ptr):
    cdef uintptr_t graph = graph_ptr
    err = gdf_add_transposed_adj_list(<gdf_graph*> graph)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

def view_transposed_adj_list(graph_ptr):
    cdef uintptr_t graph = graph_ptr
    cdef gdf_graph * g = <gdf_graph*> graph
    err = gdf_add_transposed_adj_list(g)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

    offset_col_size = g.transposedAdjList.offsets.size
    index_col_size = g.transposedAdjList.indices.size

    cdef uintptr_t offset_col_data = <uintptr_t> g.transposedAdjList.offsets.data
    cdef uintptr_t index_col_data = <uintptr_t> g.transposedAdjList.indices.data

    offsets_data = rmm.device_array_from_ptr(offset_col_data,
                                 nelem=offset_col_size,
                                 dtype=np.int32)  # ,
                                 # finalizer=rmm._make_finalizer(offset_col_data, 0))
    indices_data = rmm.device_array_from_ptr(index_col_data,
                                 nelem=index_col_size,
                                 dtype=np.int32)  # ,
                                 # finalizer=rmm._make_finalizer(index_col_data, 0))
    # g.transposedAdjList.offsets.data and g.transposedAdjList.indices.data
    # are not owned by this instance, so should not be freed here (this
    # will lead to double free, and undefined behavior).

    return cudf.Series(offsets_data), cudf.Series(indices_data)

def delete_transposed_adj_list(graph_ptr):
    """
    Delete the transposed adjacency list.
    """
    cdef uintptr_t graph = graph_ptr
    err = gdf_delete_transposed_adj_list(<gdf_graph*> graph)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

def get_two_hop_neighbors(graph_ptr):
    cdef uintptr_t graph = graph_ptr
    cdef gdf_graph * g = <gdf_graph*> graph
    cdef gdf_column c_first_col
    cdef gdf_column c_second_col
    err = gdf_get_two_hop_neighbors(g, &c_first_col, &c_second_col)
    cudf.bindings.cudf_cpp.check_gdf_error(err)
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

def number_of_vertices(graph_ptr):
    cdef uintptr_t graph = graph_ptr
    cdef gdf_graph * g = <gdf_graph*> graph
    if g.adjList:
        return g.adjList.offsets.size - 1
    elif g.transposedAdjList:
        return g.transposedAdjList.offsets.size - 1
    elif g.edgeList:
        # This code needs to be revisited when updating gdf_graph. Users
        # may expect numbrer_of_vertcies() as a cheap query but this
        # function can run for a while and also requires a significant
        # amount of additional memory. It is better to update the number
        # of vertices when creating an edge list representation.
        err = gdf_add_adj_list(g)
        cudf.bindings.cudf_cpp.check_gdf_error(err)
        return g.adjList.offsets.size - 1
    else:
        # An empty graph
        return 0

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

def _degree(graph_ptr, x=0):
    cdef uintptr_t graph = graph_ptr
    cdef gdf_graph* g = <gdf_graph*> graph

    n = number_of_vertices(graph_ptr)

    vertex_col = cudf.Series(np.zeros(n, dtype=np.int32))
    c_vertex_col = get_gdf_column_view(vertex_col)
    if g.adjList:
        err = g.adjList.get_vertex_identifiers(&c_vertex_col)
    else:
        err = g.transposedAdjList.get_vertex_identifiers(&c_vertex_col)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

    degree_col = cudf.Series(np.zeros(n, dtype=np.int32))
    cdef gdf_column c_degree_col = get_gdf_column_view(degree_col)
    err = gdf_degree(g, &c_degree_col, <int>x)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

    return vertex_col, degree_col
