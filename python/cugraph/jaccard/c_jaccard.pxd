from c_graph cimport *

cdef extern from "cugraph.h":
    
    cdef gdf_error gdf_jaccard(gdf_graph *graph, void *c_gamma, gdf_column *weights, gdf_column *weight_j)


    
