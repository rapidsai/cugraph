from c_bfs cimport *
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
import cudf
from librmm_cffi import librmm as rmm
#from pygdf import Column
import numpy as np

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
    df : cudf.DataFrame
        df['vertex'][i] gives the vertex id of the i'th vertex
        df['distance'][i] gives the path distance for the i'th vertex from the starting vertex
        df['predecessor'][i] gives for the i'th vertex the vertex it was reached from in the traversal
        
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
    
    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    cdef uintptr_t vertex_ptr = create_column(df['vertex'])
    df['distance'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    cdef uintptr_t distances_ptr = create_column(df['distance'])
    df['predecessor'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    cdef uintptr_t predecessors_ptr = create_column(df['predecessor'])
    
    err = g.adjList.get_vertex_identifiers(<gdf_column*>vertex_ptr)
    cudf.bindings.cudf_cpp.check_gdf_error(err)
    
    gdf_bfs(<gdf_graph*>g, <gdf_column*>distances_ptr, <gdf_column*>predecessors_ptr, <int>start, <bool>directed)
    return df