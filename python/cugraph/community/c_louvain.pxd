from cugraph.structure.c_graph cimport *

cdef extern from "cugraph.h":
    cdef gdf_error gdf_louvain(gdf_graph *graph, void *final_modularity, void *num_level, gdf_column *louvain_parts)
    
