from c_graph cimport *
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
import cudf
from librmm_cffi import librmm as rmm
import numpy as np

dtypes = {np.int32: GDF_INT32, np.int64: GDF_INT64, np.float32: GDF_FLOAT32, np.float64: GDF_FLOAT64}

cdef create_column(col):
    
    x= <gdf_column*>malloc(sizeof(gdf_column))
    cdef gdf_column* c_col = <gdf_column*>malloc(sizeof(gdf_column))
    cdef uintptr_t data_ptr = cudf.bindings.cudf_cpp.get_column_data_ptr(col._column)
    #cdef uintptr_t valid_ptr = cudf.bindings.cudf_cpp.get_column_valid_ptr(col._column)

    err = gdf_column_view_augmented(<gdf_column*>c_col,
                                    <void*> data_ptr,
                                    <gdf_valid_type*> 0,
                                    <gdf_size_type>len(col),
                                    dtypes[col.dtype.type],
                                    <gdf_size_type>col.null_count)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

    cdef uintptr_t col_ptr = <uintptr_t>c_col
    return col_ptr

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
        cdef gdf_graph* graph
        graph = <gdf_graph*>calloc(1,sizeof(gdf_graph))

        cdef uintptr_t graph_ptr = <uintptr_t>graph
        self.graph_ptr = graph_ptr


    def add_edge_list(self, source_col, dest_col, value_col=None):
        """
        Warp existing gdf columns representing an edge list in a gdf_graph. cuGraph does not own the memory used to represent this graph. This function does not allocate memory. 
        The cuGraph graph should not already contain the connectivity information as an edge list.
        If successful, the cuGraph graph descriptor contains the newly added edge list (edge_data is optional).
        Parameters
        ----------
        source_indices : gdf_column       
            This gdf_column of size E (number of edges) contains the index of the source for each edge.
            Indices must be in the range [0, V-1]. 
        destination_indices   : gdf_column
            This gdf_column of size E (number of edges) contains the index of the destination for each edge. 
            Indices must be in the range [0, V-1].
        edge_data (optional)  : gdf_column
            This pointer can be ``none``. If not, this gdf_column of size E (number of edges) contains the weiht for each edge. 
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
        cdef uintptr_t source=create_column(source_col)
        cdef uintptr_t dest=create_column(dest_col)
        cdef uintptr_t value
        if value_col is None:
            value = 0
        else:
            value=create_column(value_col)

        err = gdf_edge_list_view(<gdf_graph*>graph,
                                 <gdf_column*>source,
                                 <gdf_column*>dest,
                                 <gdf_column*>value)
        cudf.bindings.cudf_cpp.check_gdf_error(err)    

    def view_edge_list(self):
        """
        Display the edge list.
        """
        cdef uintptr_t graph = self.graph_ptr
        cdef gdf_graph* g = <gdf_graph*>graph
        col_size = g.edgeList.src_indices.size

        cdef uintptr_t src_col_data = <uintptr_t>g.edgeList.src_indices.data
        cdef uintptr_t dest_col_data = <uintptr_t>g.edgeList.dest_indices.data

        src_data = rmm.device_array_from_ptr(src_col_data,
                                     nelem=col_size,
                                     dtype=np.int32,
                                     finalizer=rmm._make_finalizer(src_col_data, 0))
        dest_data = rmm.device_array_from_ptr(dest_col_data,
                                     nelem=col_size,
                                     dtype=np.int32,
                                     finalizer=rmm._make_finalizer(dest_col_data, 0))

        return cudf.Series(src_data), cudf.Series(dest_data)

    def to_edge_list(self):
        """
        Compute the edge list from adjacency list and return sources and destinations as cudf Series.
        """
        cdef uintptr_t graph = self.graph_ptr
        err = gdf_add_edge_list(<gdf_graph*>graph)
        cudf.bindings.cudf_cpp.check_gdf_error(err)

        cdef gdf_graph* g = <gdf_graph*>graph
        col_size = g.edgeList.src_indices.size
        cdef uintptr_t src_col_data = <uintptr_t>g.edgeList.src_indices.data
        cdef uintptr_t dest_col_data = <uintptr_t>g.edgeList.dest_indices.data

        src_data = rmm.device_array_from_ptr(src_col_data,
                                     nelem=col_size,
                                     dtype=np.int32)#,
                                     #finalizer=rmm._make_finalizer(src_col_data, 0))
        dest_data = rmm.device_array_from_ptr(dest_col_data,
                                     nelem=col_size,
                                     dtype=np.int32)#,
                                     #finalizer=rmm._make_finalizer(dest_col_data, 0))

        return cudf.Series(src_data), cudf.Series(dest_data)

    def delete_edge_list(self):
        """
        Delete the edge list.
        """
        cdef uintptr_t graph = self.graph_ptr
        err = gdf_delete_edge_list(<gdf_graph*>graph)
        cudf.bindings.cudf_cpp.check_gdf_error(err)

    def add_adj_list(self, offsets_col, indices_col, value_col):
        """
        Warp existing gdf columns representing an adjacency list in a gdf_graph.
        """
        cdef uintptr_t graph = self.graph_ptr
        cdef uintptr_t offsets=create_column(offsets_col)
        cdef uintptr_t indices=create_column(indices_col)
        cdef uintptr_t value
        if value_col is None:
            value = 0
        else:
            value=create_column(value_col)
    
        err = gdf_adj_list_view(<gdf_graph*>graph,
                                <gdf_column*>offsets,
                                <gdf_column*>indices,
                                <gdf_column*>value)
        cudf.bindings.cudf_cpp.check_gdf_error(err)
        
    def to_adj_list(self):
        """
        Compute the adjacency list from edge list and return offsets and indices as cudf Series.
        """
        cdef uintptr_t graph = self.graph_ptr
        err = gdf_add_adj_list(<gdf_graph*>graph)
        cudf.bindings.cudf_cpp.check_gdf_error(err)
        
        cdef gdf_graph* g = <gdf_graph*>graph
        col_size_off = g.adjList.offsets.size
        col_size_ind = g.adjList.indices.size

        cdef uintptr_t offsets_col_data = <uintptr_t>g.adjList.offsets.data
        cdef uintptr_t indices_col_data = <uintptr_t>g.adjList.indices.data

        offsets_data = rmm.device_array_from_ptr(offsets_col_data,
                                     nelem=col_size_off,
                                     dtype=np.int32,
                                     finalizer=rmm._make_finalizer(offsets_col_data, 0))
        indices_data = rmm.device_array_from_ptr(indices_col_data,
                                     nelem=col_size_ind,
                                     dtype=np.int32,
                                     finalizer=rmm._make_finalizer(indices_col_data, 0))

        return cudf.Series(offsets_data), cudf.Series(indices_data)

    def delete_adj_list(self):
        """
        Delete the adjacency list.
        """
        cdef uintptr_t graph = self.graph_ptr
        err = gdf_delete_adj_list(<gdf_graph*>graph)
        cudf.bindings.cudf_cpp.check_gdf_error(err)

    def add_transpose(self):
        """
        Compute the transposed adjacency list from the edge list and add it to the existing graph.
        """
        cdef uintptr_t graph = self.graph_ptr
        err = gdf_add_transpose(<gdf_graph*>graph)
        cudf.bindings.cudf_cpp.check_gdf_error(err)

    def delete_transpose(self):
        """
        Delete the transposed adjacency list.
        """
        cdef uintptr_t graph = self.graph_ptr
        err = gdf_delete_transpose(<gdf_graph*>graph)
        cudf.bindings.cudf_cpp.check_gdf_error(err)


