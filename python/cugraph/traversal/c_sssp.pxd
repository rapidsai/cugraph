from cugraph.structure.c_graph cimport *


cdef extern from "cugraph.h":

    cdef gdf_error gdf_sssp(gdf_graph *graph, gdf_column *distances, gdf_column *predecessors, int start_vertex)
