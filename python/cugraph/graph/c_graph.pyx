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

    def __del__(self):
        cdef uintptr_t graph = self.graph_ptr
        cdef gdf_graph * g = < gdf_graph *> graph
        self.delete_edge_list()
        self.delete_adj_list()
        self.delete_transpose()
        free(g)

    def add_edge_list(self, source_col, dest_col, value_col=None):
        """
        Wrap existing gdf columns representing an edge list in a gdf_graph. cuGraph 
        does not own the memory used to represent this graph. This function does not 
        allocate memory. The cuGraph graph should not already contain the connectivity 
        information as an edge list. If successful, the cuGraph graph descriptor 
        contains the newly added edge list (edge_data is optional).
        Parameters
        ----------
        source_indices : gdf_column       
            This gdf_column of size E (number of edges) contains the index of the 
                source for each edge.
            Indices must be in the range [0, V-1]. 
        destination_indices   : gdf_column
            This gdf_column of size E (number of edges) contains the index of the 
                destination for each edge. 
            Indices must be in the range [0, V-1].
        edge_data (optional)  : gdf_column
            This pointer can be ``none``. If not, this gdf_column of size E 
                (number of edges) contains the weight for each edge. 
            The type expected to be floating point.
        Examples
        --------
        >>> import cuGraph
        >>> import cudf
        >>> from scipy.io import mmread
        >>> M = ReadMtxFile(graph_file)
        >>> sources = cudf.Series(M.row)
        >>> destinations = cudf.Series(M.col)
        >>> G = cuGraph.Graph()
        >>> G.add_edge_list(sources,destinations,none)
        """
        cdef uintptr_t graph = self.graph_ptr
        cdef uintptr_t source = create_column(source_col)
        cdef uintptr_t dest = create_column(dest_col)
        cdef uintptr_t value
        if value_col is None:
            value = 0
        else:
            value = create_column(value_col)

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

    def add_adj_list(self, offsets_col, indices_col, value_col):
        """
        Warp existing gdf columns representing an adjacency list in a gdf_graph.
        """
        cdef uintptr_t graph = self.graph_ptr
        cdef uintptr_t offsets = create_column(offsets_col)
        cdef uintptr_t indices = create_column(indices_col)
        cdef uintptr_t value
        if value_col is None:
            value = 0
        else:
            value = create_column(value_col)

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
        Compute the adjacency list from edge list and return offsets and indices as cudf Series.
        """
        cdef uintptr_t graph = self.graph_ptr
        cdef gdf_graph * g = < gdf_graph *> graph
        err = gdf_add_adj_list(g)
        cudf.bindings.cudf_cpp.check_gdf_error(err)

        col_size_off = g.adjList.offsets.size
        col_size_ind = g.adjList.indices.size

        cdef uintptr_t offsets_col_data = < uintptr_t > g.adjList.offsets.data
        cdef uintptr_t indices_col_data = < uintptr_t > g.adjList.indices.data

        offsets_data = rmm.device_array_from_ptr(offsets_col_data,
                                     nelem=col_size_off,
                                     dtype=np.int32)  # ,
                                     # finalizer=rmm._make_finalizer(offsets_col_data, 0))
        indices_data = rmm.device_array_from_ptr(indices_col_data,
                                     nelem=col_size_ind,
                                     dtype=np.int32)  # ,
                                     # finalizer=rmm._make_finalizer(indices_col_data, 0))
        # g.adjList.offsets.data and g.adjList.indices.data are not owned by
        # this instance, so should not be freed here (this will lead to double
        # free, and undefined behavior).

        return cudf.Series(offsets_data), cudf.Series(indices_data)
    
    def view_transpose_adj_list(self):
        """
        Return a view of the transposed adjacency list, computing it if it doesn't
        exist.
        """
        cdef uintptr_t graph = self.graph_ptr
        cdef gdf_graph * g = < gdf_graph *> graph
        err = gdf_add_transpose(g)
        cudf.bindings.cudf_cpp.check_gdf_error(err)
        
        off_size = g.transposedAdjList.offsets.size
        ind_size = g.transposedAdjList.indices.size
        
        cdef uintptr_t offsets_col_data = < uintptr_t > g.transposedAdjList.offsets.data
        cdef uintptr_t indices_col_data = < uintptr_t > g.transposedAdjList.indices.data
        
        offsets_data = rmm.device_array_from_ptr(offsets_col_data,
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

    def add_transpose(self):
        """
        Compute the transposed adjacency list from the edge list and add it to the existing graph.
        """
        cdef uintptr_t graph = self.graph_ptr
        err = gdf_add_transpose(< gdf_graph *> graph)
        cudf.bindings.cudf_cpp.check_gdf_error(err)

    def delete_transpose(self):
        """
        Delete the transposed adjacency list.
        """
        cdef uintptr_t graph = self.graph_ptr
        err = gdf_delete_transpose(< gdf_graph *> graph)
        cudf.bindings.cudf_cpp.check_gdf_error(err)
