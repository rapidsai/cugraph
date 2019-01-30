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