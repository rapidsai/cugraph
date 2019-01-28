from c_bfs cimport *
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
import cudf
from librmm_cffi import librmm as rmm
#from pygdf import Column
import numpy as np

dtypes = {np.int32: GDF_INT32, np.int64: GDF_INT64, np.float32: GDF_FLOAT32, np.float64: GDF_FLOAT64}

def _get_ctype_ptr(obj):
    # The manner to access the pointers in the gdf's might change, so
    # encapsulating access in the following 3 methods. They might also be
    # part of future gdf versions.
    return obj.device_ctypes_pointer.value

def _get_column_data_ptr(obj):
    return _get_ctype_ptr(obj._column._data.to_gpu_array())

def _get_column_valid_ptr(obj):
    return _get_ctype_ptr(obj._column._mask.to_gpu_array())

#def _get_gdf_as_matrix_ptr(gdf):
#    return self._get_ctype_ptr(gdf.as_gpu_matrix())

cdef create_column(col):
    
    x= <gdf_column*>malloc(sizeof(gdf_column))
    cdef gdf_column* c_col = <gdf_column*>malloc(sizeof(gdf_column))
    cdef uintptr_t data_ptr = _get_column_data_ptr(col)
    #cdef uintptr_t valid_ptr = _get_column_valid_ptr(col)

    gdf_column_view_augmented(<gdf_column*>c_col,
                              <void*> data_ptr,
                              <gdf_valid_type*> 0,
                              <gdf_size_type>len(col),
                              dtypes[col.dtype.type],
                              <gdf_size_type>col.null_count)
    
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

        gdf_edge_list_view(<gdf_graph*>graph,
                       <gdf_column*>source,
                       <gdf_column*>dest,
                       <gdf_column*>value)
    
    def view_edge_list(self):
        ##TO DO
        """
        Display the edge list.
        """
        cdef uintptr_t graph = self.graph_ptr
        cdef gdf_graph* g = <gdf_graph*>graph
        size = g.edgeList.src_indices.size
        print(size)
        cdef object cffi_view = <object>g.edgeList.src_indices
        data = cudf._gdf.cffi_view_to_column_mem(cffi_view)
        #return pygdf.Series(data)        
        return 0

    def add_adj_list(self, offsets_col, indices_col, value_col):
        """
        Warp existing gdf columns representing an adjacency list in a gdf_graph.
        """
        ##TO TEST
        cdef uintptr_t graph = self.graph_ptr
        cdef uintptr_t offsets=create_column(offsets_col)
        cdef uintptr_t indices=create_column(indices_col)
        cdef uintptr_t value
        if value_col is None:
            value = 0
        else:
            value=create_column(value_col)
    
        gdf_adj_list_view(<gdf_graph*>graph,
                       <gdf_column*>offsets,
                       <gdf_column*>indices,
                       <gdf_column*>value)


    def add_transpose(self):
        """
        Compute the transposed adjacency list from the edge list and add it to the existing graph.
        """
        cdef uintptr_t graph = self.graph_ptr
        gdf_add_transpose(<gdf_graph*>graph)

cpdef bfs(G, start, directed=True):
    """
    Find the distances and predecessors for a breadth first traversal of a graph.
    
    Parameters
    ----------
    G : cugraph.graph
        cuGraph graph descriptor, should contain the connectivity information as an
        adjacency list.
    start : Integer
        The index of the graph vertex from which the traversal begins
    directed : bool
        Indicates whether the graph in question is a directed graph, or whether
        each edge has a corresponding reverse edge. (Allows optimizations if the
        graph is undirected)
    
    Returns
    -------
    distances, predecessors : cudf.Series
        distances gives the path distance for each vertex from the starting vertex
        predecessors gives for each vertex the vertex it was reached from in the traversal
        
    Examples
    --------
    >>> M = ReadMtxFile(graph_file)
    >>> sources = cudf.Series(M.row)
    >>> destinations = cudf.Series(M.col)
    >>> G = cuGraph.Graph()
    >>> G.add_edge_list(sources,destinations,none)
    >>> dist, pred = cuGraph.bfs(G, 0, false)
    """
    
    cdef uintptr_t graph = G.graph_ptr
    cdef gdf_graph* g = <gdf_graph*>graph
    num_verts = g.adjList.offsets.size - 1
    distances = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    cdef uintptr_t distances_ptr = create_column(distances)
    predecessors = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    cdef uintptr_t predecessors_ptr = create_column(distances)
    
    gdf_bfs(<gdf_graph*>g, <gdf_column*>distances_ptr, <gdf_column*>predecessors_ptr, <int>start, <bool>directed)
    return distances, predecessors