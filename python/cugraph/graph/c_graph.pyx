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


cdef gdf_column get_gdf_column_view(col):
    """
    This function returns a C++ gdf_column object from the Python cudf Series
    object by shallow copying. The returned C++ object is expected to be used
    as a temporary variable to pass the column data encapsulated in the Python
    cudf Series object to C++ functions expecting (pointers to) C++ gdf_column
    objects. It is the caller's responsibility to insure that col out-lives the
    returned view object. cudf has column_view_from_column and using this is,
    in general, better design than creating our own, but we will keep this as
    cudf is planning to remove the function. cudf plans to redesign
    cudf::column to fundamentally solve this problem, so once they finished the
    redesign, we need to update this code to use their new features. Until that
    time, we may rely on this as a temporary solution.
    """
    cdef gdf_column c_col
    cdef uintptr_t data_ptr = cudf.bindings.cudf_cpp.get_column_data_ptr(col._column)
    cdef uintptr_t valid_ptr
    if col._column._mask is None:
        valid_ptr = 0
    else:
        valid_ptr = cudf.bindings.cudf_cpp.get_column_valid_ptr(col._column)
    cdef gdf_dtype_extra_info c_extra_dtype_info = gdf_dtype_extra_info(time_unit=TIME_UNIT_NONE)

    err = gdf_column_view_augmented(<gdf_column*> &c_col,
                                    <void*> data_ptr,
                                    <gdf_valid_type*> valid_ptr,
                                    <gdf_size_type> len(col),
                                    dtypes[col.dtype.type],
                                    <gdf_size_type> col.null_count,
                                    c_extra_dtype_info)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

    return c_col


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
        g = <gdf_graph*> calloc(1, sizeof(gdf_graph))

        cdef uintptr_t graph_ptr = <uintptr_t> g
        self.graph_ptr = graph_ptr

        self.edge_list_source_col = None
        self.edge_list_dest_col = None
        self.edge_list_value_col = None

        self.adj_list_offset_col = None
        self.adj_list_index_col = None
        self.adj_list_value_col = None

    def __del__(self):
        cdef uintptr_t graph = self.graph_ptr
        cdef gdf_graph * g = <gdf_graph*> graph
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
            self.edge_list_source_col = source_col
            self.edge_list_dest_col = dest_col
            self.edge_list_value_col = value_col
        else:
            self.edge_list_source_col = source_col.copy()
            self.edge_list_dest_col = dest_col.copy()
            self.edge_list_value_col = value_col.copy()

        cdef uintptr_t graph = self.graph_ptr
        cdef gdf_column c_source_col = get_gdf_column_view(self.edge_list_source_col)
        cdef gdf_column c_dest_col = get_gdf_column_view(self.edge_list_dest_col)
        cdef gdf_column c_value_col
        cdef gdf_column * c_value_col_ptr
        if value_col is None:
            c_value_col_ptr = NULL
        else:
            c_value_col = get_gdf_column_view(self.edge_list_value_col)
            c_value_col_ptr = &c_value_col

        err = gdf_edge_list_view(<gdf_graph*> graph,
                                 &c_source_col,
                                 &c_dest_col,
                                 c_value_col_ptr)
        cudf.bindings.cudf_cpp.check_gdf_error(err)

    def view_edge_list(self):
        """
        Display the edge list. Compute it if needed.
        """
        cdef uintptr_t graph = self.graph_ptr
        cdef gdf_graph * g = <gdf_graph*> graph
        err = gdf_add_edge_list(g)
        cudf.bindings.cudf_cpp.check_gdf_error(err)

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

    def delete_edge_list(self):
        """
        Delete the edge list.
        """
        cdef uintptr_t graph = self.graph_ptr
        err = gdf_delete_edge_list(<gdf_graph*> graph)
        cudf.bindings.cudf_cpp.check_gdf_error(err)

        # decrease reference count to free memory if the referenced objects are
        # no longer used.
        self.edge_list_source_col = None
        self.edge_list_dest_col = None
        self.edge_list_value_col = None

    def add_adj_list(self, offset_col, index_col, value_col=None, copy=False):
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
            self.adj_list_offset_col = offset_col
            self.adj_list_index_col = index_col
            self.adj_list_value_col = value_col
        else:
            self.adj_list_offset_col = offset_col.copy()
            self.adj_list_index_col = index_col.copy()
            self_adj_list_value_col = value_col.copy()

        cdef uintptr_t graph = self.graph_ptr
        cdef gdf_column c_offset_col = get_gdf_column_view(self.adj_list_offset_col)
        cdef gdf_column c_index_col = get_gdf_column_view(self.adj_list_index_col)
        cdef gdf_column c_value_col
        cdef gdf_column * c_value_col_ptr
        if value_col is None:
            c_value_col_ptr = NULL
        else:
            c_value_col = get_gdf_column_view(self.adj_list_value_col)
            c_value_col_ptr = &c_value_col

        err = gdf_adj_list_view(<gdf_graph*> graph,
                                &c_offset_col,
                                &c_index_col,
                                c_value_col_ptr)
        cudf.bindings.cudf_cpp.check_gdf_error(err)

    def view_adj_list(self):
        """
        Display the adjacency list. Compute it if needed.
        """
        cdef uintptr_t graph = self.graph_ptr
        cdef gdf_graph * g = <gdf_graph*> graph
        err = gdf_add_adj_list(g)
        cudf.bindings.cudf_cpp.check_gdf_error(err)

        offset_col_size = self.number_of_vertices() + 1
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

    def delete_adj_list(self):
        """
        Delete the adjacency list.
        """
        cdef uintptr_t graph = self.graph_ptr
        err = gdf_delete_adj_list(<gdf_graph*> graph)
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
        err = gdf_add_transposed_adj_list(<gdf_graph*> graph)
        cudf.bindings.cudf_cpp.check_gdf_error(err)

    def view_transposed_adj_list(self):
        """
        Display the transposed adjacency list. Compute it if needed.
        """
        cdef uintptr_t graph = self.graph_ptr
        cdef gdf_graph * g = <gdf_graph*> graph
        err = gdf_add_transposed_adj_list(g)
        cudf.bindings.cudf_cpp.check_gdf_error(err)

        offset_col_size = self.number_of_vertices() + 1
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

    def delete_transposed_adj_list(self):
        """
        Delete the transposed adjacency list.
        """
        cdef uintptr_t graph = self.graph_ptr
        err = gdf_delete_transposed_adj_list(<gdf_graph*> graph)
        cudf.bindings.cudf_cpp.check_gdf_error(err)

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

    def number_of_vertices(self):
        """
        Get the number of vertices in the graph
        """
        cdef uintptr_t graph = self.graph_ptr
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

    def in_degree(self, vertex_subset = None):
        """
        Calculates and returns the in-degree of vertices. Vertex in-degree
        is the number of edges pointing in to the vertex.
        Parameters
        ----------
        vertex_subset(optional, default=all vertices) : cudf.Series or iterable container
            A container of vertices for displaying corresponding in-degree
        Returns
        -------
        df  : cudf.DataFrame
        GPU data frame of size N (the default) or the size of the given vertices (vertex_subset)
        containing the in_degree. The ordering is relative to the adjacency list, or that
        given by the specified vertex_subset.

        df['vertex']: The vertex IDs (will be identical to vertex_subset if specified)
        df['degree']: The computed in-degree of the corresponding vertex
        Examples
        --------
        >>> import numpy as np
        >>> import pytest
        >>> from scipy.io import mmread
        >>>
        >>> import cudf
        >>> import cugraph
        >>> mm_file = '/datasets/networks/karate.mtx'
        >>> M = mmread(mm_file).asfptype()
        >>> sources = cudf.Series(M.row)
        >>> destinations = cudf.Series(M.col)
        >>>
        >>> G = cugraph.Graph()
        >>> G.add_edge_list(sources, destinations)
        >>> in_degree_df = G.in_degree([0,9,12])
        """
        return self._degree(vertex_subset , x=1)

    def out_degree(self, vertex_subset = None):
        """
        Calculates and returns the out-degree of vertices. Vertex out-degree
        is the number of edges pointing out from the vertex.
        Parameters
        ----------
        vertex_subset(optional, default=all vertices) : cudf.Series or iterable container
            A container of vertices for displaying corresponding out-degree
        Returns
        -------
        df  : cudf.DataFrame
        GPU data frame of size N (the default) or the size of the given vertices (vertex_subset)
        containing the out_degree. The ordering is relative to the adjacency list, or that
        given by the specified vertex_subset.

        df['vertex']: The vertex IDs (will be identical to vertex_subset if specified)
        df['degree']: The computed out-degree of the corresponding vertex
        Examples
        --------
        >>> import numpy as np
        >>> import pytest
        >>> from scipy.io import mmread
        >>>
        >>> import cudf
        >>> import cugraph
        >>> mm_file = '/datasets/networks/karate.mtx'
        >>> M = mmread(mm_file).asfptype()
        >>> sources = cudf.Series(M.row)
        >>> destinations = cudf.Series(M.col)
        >>>
        >>> G = cugraph.Graph()
        >>> G.add_edge_list(sources, destinations)
        >>> out_degree_df = G.out_degree([0,9,12])
        """
        return self._degree(vertex_subset, x=2)

    def degree(self, vertex_subset = None):
        """
        Calculates and returns the degree of vertices. Vertex degree
        is the number of edges adjacent to that vertex.
        Parameters
        ----------
        vertex_subset(optional, default=all vertices) : cudf.Series or iterable container
            A container of vertices for displaying corresponding degree
        Returns
        -------
        df  : cudf.DataFrame
        GPU data frame of size N (the default) or the size of the given vertices (vertex_subset)
        containing the degree. The ordering is relative to the adjacency list, or that
        given by the specified vertex_subset.

        df['vertex']: The vertex IDs (will be identical to vertex_subset if specified)
        df['degree']: The computed degree of the corresponding vertex
        Examples
        --------
        >>> import numpy as np
        >>> import pytest
        >>> from scipy.io import mmread
        >>>
        >>> import cudf
        >>> import cugraph
        >>> mm_file = '/datasets/networks/karate.mtx'
        >>> M = mmread(mm_file).asfptype()
        >>> sources = cudf.Series(M.row)
        >>> destinations = cudf.Series(M.col)
        >>>
        >>> G = cugraph.Graph()
        >>> G.add_edge_list(sources, destinations)
        >>> degree_df = G.degree([0,9,12])
        """
        return self._degree(vertex_subset)

    def _degree(self, vertex_subset, x = 0):
        cdef uintptr_t graph = self.graph_ptr
        cdef gdf_graph* g = <gdf_graph*> graph

        n = self.number_of_vertices()

        df = cudf.DataFrame()
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

        if vertex_subset is None:
            df['vertex'] = vertex_col
            df['degree'] = degree_col
        else:
            df['vertex'] = cudf.Series(np.asarray(vertex_subset, dtype=np.int32))
            df['degree'] = cudf.Series(np.asarray([degree_col[i] for i in vertex_subset], dtype=np.int32))
            del vertex_col
            del degree_col

        return df
