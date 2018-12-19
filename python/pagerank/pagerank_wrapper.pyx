from c_pagerank cimport *
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


cpdef pagerank(G,alpha=0.85, max_iter=100, tol=1.0e-5):
    """
    Find the PageRank vertex values for a graph. cuGraph computes an approximation of the Pagerank eigenvector using the power method. 
    The number of iterations depends on the properties of the network itself; it increases when the tolerance descreases and/or alpha increases toward the limiting value of 1.
    The user is free to use default values or to provide inputs for the initial guess, tolerance and maximum number of iterations.
    
    Parameters
    ----------
    graph : cuGraph.Graph                  
       cuGraph graph descriptor, should contain the connectivity information as an edge list (edge weights are not used for this algorithm). 
       The transposed adjacency list will be computed if not already present.
    alpha : float                  
       The damping factor alpha represents the probability to follow an outgoing edge, standard value is 0.85. 
       Thus, 1.0-alpha is the probability to “teleport” to a random node. Alpha should be greater than 0.0 and strictly lower than 1.0.
    tolerance : float              
       Set the tolerance the approximation, this parameter should be a small magnitude value. 
       The lower the tolerance the better the approximation. If this value is 0.0f, cuGraph will use the default value which is 1.0E-5. 
       Setting too small a tolerance can lead to non-convergence due to numerical roundoff. Usually values between 0.01 and 0.00001 are acceptable.
    max_iter  : int                
       The maximum number of iterations before an answer is returned. This can be used to limit the execution time and do an early exit before the solver reaches the convergence tolerance. 
       If this value is lower or equal to 0 cuGraph will use the default value, which is 100.
    
    Returns
    -------
    PageRank : cudf.Series
        pagerank[i] is the PageRank of vertex i.

    Examples
    --------
    >>> M = ReadMtxFile(graph_file)
    >>> sources = cudf.Series(M.row)
    >>> destinations = cudf.Series(M.col)
    >>> G = cuGraph.Graph()
    >>> G.add_edge_list(sources,destinations,None)
    >>> pr = cuGraph.pagerank(G, alpha = 0.85, max_iter = 500, tol = 1.0e-05)
    """
    cdef uintptr_t graph = G.graph_ptr
    gdf_add_transpose(<gdf_graph*>graph)
    
    cdef gdf_graph* g = <gdf_graph*>graph
    size = g.transposedAdjList.offsets.size
    pagerank = cudf.Series(np.zeros(g.transposedAdjList.offsets.size,dtype=np.float64))
    cdef uintptr_t pagerank_ptr = create_column(pagerank)    

    gdf_pagerank(<gdf_graph*>graph, <gdf_column*>pagerank_ptr, <float> alpha, <float> tol, <int> max_iter, <bool> 0)
    return pagerank

