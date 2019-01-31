from c_graph cimport *
from libcpp cimport bool

cdef extern from "cugraph.h":

    cdef gdf_error gdf_bfs(gdf_graph *graph, gdf_column *distances, gdf_column *predecessors, int start_node, bool directed)
