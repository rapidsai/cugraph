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

from c_graph cimport *
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
import cudf
from librmm_cffi import librmm as rmm
import numpy as np


dtypes = {np.int32: GDF_INT32, np.int64: GDF_INT64, np.float32: GDF_FLOAT32, np.float64: GDF_FLOAT64}


cdef create_column(col):
    cdef gdf_column * c_col = < gdf_column *> malloc(sizeof(gdf_column))
    cdef uintptr_t data_ptr = cudf.bindings.cudf_cpp.get_column_data_ptr(col._column)
    # cdef uintptr_t valid_ptr = cudf.bindings.cudf_cpp.get_column_valid_ptr(col._column)
    cdef gdf_dtype_extra_info c_extra_dtype_info = gdf_dtype_extra_info(time_unit=TIME_UNIT_NONE)


    err = gdf_column_view_augmented(< gdf_column *> c_col,
                                    < void *> data_ptr,
                                    < gdf_valid_type *> 0,
                                    < gdf_size_type > len(col),
                                    dtypes[col.dtype.type],
                                    < gdf_size_type > col.null_count,
                                    c_extra_dtype_info)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

    cdef uintptr_t col_ptr = < uintptr_t > c_col
    return col_ptr


cdef delete_column(col_ptr):
    cdef uintptr_t col = col_ptr
    cdef gdf_column * c_col = < gdf_column *> col
    free(c_col)
    return


class Graph:
    """
    cuGraph graph class containing basic graph creation and transformation operations.
    """
    def __init__(self):
        """
        Returns
        -------
        Graph : cuGraph.Graph.
        Examples
        --------
        >>> import cuGraph
        >>> G = cuGraph.Graph()
        """
        cdef gdf_graph * g
        g = < gdf_graph *> calloc(1, sizeof(gdf_graph))

        cdef uintptr_t graph_ptr = < uintptr_t > g
        self.graph_ptr = graph_ptr

        self.edge_list_source_col = None
        self.edge_list_dest_col = None
        self.edge_list_value_col = None

        self.adj_list_offset_col = None
        self.adj_list_index_col = None
        self.adj_list_value_col = None

    def __del__(self):
        cdef uintptr_t graph = self.graph_ptr
        cdef gdf_graph * g = < gdf_graph *> graph
        self.delete_edge_list()
        self.delete_adj_list()
        self.delete_transposed_adj_list()
        free(g)

    def add_edge_list(self, source_col, dest_col, value_col=None, copy=False):
        """
        Create the edge list representation of a Graph. The passed source_col
        and dest_col arguments wrap gdf_column objects that represent a graph
        using the edge list format. If value_col is None, an unweighted graph
        is created. If value_col is not None, an weighted graph is created. If
        copy is False, this function stores references to the passed objects
        pointed by source_col and dest_col. If copy is True, this funcion
        stores references to the deep-copies of the passed objects pointed by
        source_col and dest_col. If this class instance already stores a graph,
        invoking this function raises an error.
        Parameters
        ----------
        source_col : cudf.Series
            This cudf.Series wraps a gdf_column of size E (E: number of edges).
            The gdf column contains the source index for each edge.
            Source indices must be in the range [0, V) (V: number of vertices).
        dest_col : cudf.Series
            This cudf.Series wraps a gdf_column of size E (E: number of edges).
            The gdf column contains the destination index for each edge.
            Destination indices must be in the range [0, V) (V: number of
            vertices).
        value_col (optional) : cudf.Series
            This pointer can be ``none``.
            If not, this cudf.Series wraps a gdf_column of size E (E: number of
            edges).
            The gdf column contains the weight value for each edge.
            The expected type of the gdf_column element is floating point
            number.
        Examples
        --------
        >>> import numpy as np
        >>> import pytest
        >>> from scipy.io import mmread
        >>>
        >>> import cudf
        >>> import cugraph
        >>>
        >>>
        >>> mm_file = '../datasets/karate.mtx'
        >>> M = mmread(mm_file).asfptype()
        >>> sources = cudf.Series(M.row)
        >>> destinations = cudf.Series(M.col)
        >>>
        >>> G = cugraph.Graph()
        >>> G.add_edge_list(sources, destinations, None)
        """
        # If copy is False, increase the reference count of the Python objects
        # referenced by the input arguments source_col, dest_col, and value_col
        # (if not None) to avoid garbage collection while they are still in use
        # inside this class. If copy is set to True, deep-copy the objects.
        if copy is False:
            self.edge_list_source_col = source_col;
            self.edge_list_dest_col = dest_col;
            self.edge_list_value_col = value_col;
        else:
            self.edge_list_source_col = source_col.copy();
            self.edge_list_dest_col = dest_col.copy();
            self_edge_list_value_col = value_col.copy();

        cdef uintptr_t graph = self.graph_ptr
        cdef uintptr_t source = create_column(self.edge_list_source_col)
        cdef uintptr_t dest = create_column(self.edge_list_dest_col)
        cdef uintptr_t value
        if value_col is None:
            value = 0
        else:
            value = create_column(self.edge_list_value_col)

        try:
            err = gdf_edge_list_view(< gdf_graph *> graph,
                                     < gdf_column *> source,
                                     < gdf_column *> dest,
                                     < gdf_column *> value)
            cudf.bindings.cudf_cpp.check_gdf_error(err)
        finally:
            delete_column(source)
            delete_column(dest)
            if value is not 0:
                delete_column(value)

    def num_vertices(self):
        """
        Get the number of vertices in the graph
        """
        cdef uintptr_t graph = self.graph_ptr
        cdef gdf_graph * g = < gdf_graph *> graph
        err = gdf_add_adj_list(g)
        cudf.bindings.cudf_cpp.check_gdf_error(err)
        return g.adjList.offsets.size - 1   

    def view_edge_list(self):
        """
        Display the edge list. Compute it if needed.
        """
        cdef uintptr_t graph = self.graph_ptr
        cdef gdf_graph * g = < gdf_graph *> graph
        err = gdf_add_edge_list(g)
        cudf.bindings.cudf_cpp.check_gdf_error(err)

        col_size = g.edgeList.src_indices.size

        cdef uintptr_t src_col_data = < uintptr_t > g.edgeList.src_indices.data
        cdef uintptr_t dest_col_data = < uintptr_t > g.edgeList.dest_indices.data

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

    def delete_edge_list(self):
        """
        Delete the edge list.
        """
        cdef uintptr_t graph = self.graph_ptr
        err = gdf_delete_edge_list(< gdf_graph *> graph)
        cudf.bindings.cudf_cpp.check_gdf_error(err)

        # decrease reference count to free memory if the referenced objects are
        # no longer used.
        self.edge_list_source_col = None
        self.edge_list_dest_col = None
        self.edge_list_value_col = None

    def add_adj_list(self, offset_col, index_col, value_col, copy=False):
        """
        Create the adjacency list representation of a Graph. The passed
        offset_col and index_col arguments wrap gdf_column objects that
        represent a graph using the adjacency list format. If value_col is
        None, an unweighted graph is created. If value_col is not None, an
        weighted graph is created. If copy is False, this function stores
        references to the passed objects pointed by offset_col and index_col.
        If copy is True, this funcion stores references to the deep-copies of
        the passed objects pointed by offset_col and index_col. If this class
        instance already stores a graph, invoking this function raises an
        error.
        Parameters
        ----------
        offset_col : cudf.Series
            This cudf.Series wraps a gdf_column of size V + 1 (V: number of
            vertices).
            The gdf column contains the offsets for the vertices in this graph.
            Offsets must be in the range [0, E] (E: number of edges).
        index_col : cudf.Series
            This cudf.Series wraps a gdf_column of size E (E: number of edges).
            The gdf column contains the destination index for each edge.
            Destination indices must be in the range [0, V) (V: number of
            vertices).
        value_col (optional) : cudf.Series
            This pointer can be ``none``.
            If not, this cudf.Series wraps a gdf_column of size E (E: number of
            edges).
            The gdf column contains the weight value for each edge.
            The expected type of the gdf_column element is floating point
            number.
        Examples
        --------
        >>> import numpy as np
        >>> import pytest
        >>> from scipy.io import mmread
        >>>
        >>> import cudf
        >>> import cugraph
        >>>
        >>>
        >>> mm_file = '../datasets/karate.mtx'
        >>> M = mmread(mm_file).asfptype()
        >>> M = M.tocsr()
        >>> offsets = cudf.Series(M.indptr)
        >>> indices = cudf.Series(M.indices)
        >>>
        >>> G = cugraph.Graph()
        >>> G.add_adj_list(offsets, indices, None)
        """
        # If copy is False, increase the reference count of the Python objects
        # referenced by the input arguments offset_col, index_col, and
        # value_col (if not None) to avoid garbage collection while they are
        # still in use inside this class. If copy is set to True, deep-copy the
        # objects.
        if copy is False:
            self.adj_list_offset_col = offset_col;
            self.adj_list_index_col = index_col;
            self.adj_list_value_col = value_col;
        else:
            self.adj_list_offset_col = offset_col.copy();
            self.adj_list_index_col = index_col.copy();
            self_adj_list_value_col = value_col.copy();

        cdef uintptr_t graph = self.graph_ptr
        cdef uintptr_t offsets = create_column(self.adj_list_offset_col)
        cdef uintptr_t indices = create_column(self.adj_list_index_col)
        cdef uintptr_t value
        if value_col is None:
            value = 0
        else:
            value = create_column(self.adj_list_value_col)

        try:
            err = gdf_adj_list_view(< gdf_graph *> graph,
                                    < gdf_column *> offsets,
                                    < gdf_column *> indices,
                                    < gdf_column *> value)
            cudf.bindings.cudf_cpp.check_gdf_error(err)
        finally:
            delete_column(offsets)
            delete_column(indices)
            if value is not 0:
                delete_column(value)

    def view_adj_list(self):
        """
        Display the adjacency list. Compute it if needed.
        """
        cdef uintptr_t graph = self.graph_ptr
        cdef gdf_graph * g = < gdf_graph *> graph
        err = gdf_add_adj_list(g)
        cudf.bindings.cudf_cpp.check_gdf_error(err)

        col_size_off = g.adjList.offsets.size
        col_size_ind = g.adjList.indices.size

        cdef uintptr_t offset_col_data = < uintptr_t > g.adjList.offsets.data
        cdef uintptr_t index_col_data = < uintptr_t > g.adjList.indices.data

        offsets_data = rmm.device_array_from_ptr(offset_col_data,
                                     nelem=col_size_off,
                                     dtype=np.int32) # ,
                                     # finalizer=rmm._make_finalizer(offset_col_data, 0))
        indices_data = rmm.device_array_from_ptr(index_col_data,
                                     nelem=col_size_ind,
                                     dtype=np.int32) # ,
                                     # finalizer=rmm._make_finalizer(index_col_data, 0))
        # g.adjList.offsets.data and g.adjList.indices.data are not owned by
        # this instance, so should not be freed here (this will lead to double
        # free, and undefined behavior).

        return cudf.Series(offsets_data), cudf.Series(indices_data)

    def delete_adj_list(self):
        """
        Delete the adjacency list.
        """
        cdef uintptr_t graph = self.graph_ptr
        err = gdf_delete_adj_list(< gdf_graph *> graph)
        cudf.bindings.cudf_cpp.check_gdf_error(err)

        # decrease reference count to free memory if the referenced objects are
        # no longer used.
        self.adj_list_offset_col = None
        self.adj_list_index_col = None
        self.adj_list_value_col = None

    def add_transposed_adj_list(self):
        """
        Compute the transposed adjacency list from the edge list and add it to
        the existing graph.
        """
        cdef uintptr_t graph = self.graph_ptr
        err = gdf_add_transposed_adj_list(< gdf_graph *> graph)
        cudf.bindings.cudf_cpp.check_gdf_error(err)

    def view_transposed_adj_list(self):
        """
        Display the transposed adjacency list. Compute it if needed.
        """
        cdef uintptr_t graph = self.graph_ptr
        cdef gdf_graph * g = < gdf_graph *> graph
        err = gdf_add_transposed_adj_list(g)
        cudf.bindings.cudf_cpp.check_gdf_error(err)

        off_size = g.transposedAdjList.offsets.size
        ind_size = g.transposedAdjList.indices.size

        cdef uintptr_t offset_col_data = < uintptr_t > g.transposedAdjList.offsets.data
        cdef uintptr_t indices_col_data = < uintptr_t > g.transposedAdjList.indices.data

        offsets_data = rmm.device_array_from_ptr(offset_col_data,
                                     nelem=off_size,
                                     dtype=np.int32)  # ,
                                     # finalizer=rmm._make_finalizer(offsets_col_data, 0))
        indices_data = rmm.device_array_from_ptr(indices_col_data,
                                     nelem=ind_size,
                                     dtype=np.int32)  # ,
                                     # finalizer=rmm._make_finalizer(indices_col_data, 0))
        # g.transposedAdjList.offsets.data and g.transposedAdjList.indices.data
        # are not owned by this instance, so should not be freed here (this
        # will lead to double free, and undefined behavior).

        return cudf.Series(offsets_data), cudf.Series(indices_data)


    def get_two_hop_neighbors(self):
        """
        Return a dataframe containing vertex pairs such that each pair of vertices is 
        connected by a path of two hops in the graph. The resulting pairs are 
        returned in sorted order.
        
        Returns:
        df : a cudf.DataFrame object
        df['first'] the first vertex id of a pair
        df['second'] the second vertex id of a pair
        """
        cdef uintptr_t graph = self.graph_ptr
        cdef gdf_graph * g = < gdf_graph *> graph
        cdef gdf_column * first = < gdf_column *> malloc(sizeof(gdf_column))
        cdef gdf_column * second = < gdf_column *> malloc(sizeof(gdf_column))
        err = gdf_get_two_hop_neighbors(g, first, second)
        cudf.bindings.cudf_cpp.check_gdf_error(err)
        df = cudf.DataFrame()
        if first.dtype == GDF_INT32:
            first_out = rmm.device_array_from_ptr(<uintptr_t>first.data, 
                                                  nelem=first.size, 
                                                  dtype=np.int32)
            second_out = rmm.device_array_from_ptr(<uintptr_t>second.data, 
                                                   nelem=second.size, 
                                                   dtype=np.int32)
            df['first'] = first_out
            df['second'] = second_out
        if first.dtype == GDF_INT64:
            first_out = rmm.device_array_from_ptr(<uintptr_t>first.data, 
                                                  nelem=first.size, 
                                                  dtype=np.int64)
            second_out = rmm.device_array_from_ptr(<uintptr_t>second.data, 
                                                   nelem=second.size, 
                                                   dtype=np.int64)
            df['first'] = first_out
            df['second'] = second_out

        delete_column(<uintptr_t>first)
        delete_column(<uintptr_t>second)
        return df

    def delete_adj_list(self):
        """
        Delete the adjacency list.
        """
        cdef uintptr_t graph = self.graph_ptr
        err = gdf_delete_adj_list(< gdf_graph *> graph)
        cudf.bindings.cudf_cpp.check_gdf_error(err)

    def delete_transposed_adj_list(self):
        """
        Delete the transposed adjacency list.
        """
        cdef uintptr_t graph = self.graph_ptr
        err = gdf_delete_transposed_adj_list(< gdf_graph *> graph)
        cudf.bindings.cudf_cpp.check_gdf_error(err)

    def num_vertices(self):
        """
        Get the number of vertices in the graph
        """
        cdef uintptr_t graph = self.graph_ptr
        cdef gdf_graph* g = < gdf_graph *> graph
        err = gdf_add_adj_list(g)
        cudf.bindings.cudf_cpp.check_gdf_error(err)
        return g.adjList.offsets.size - 1
