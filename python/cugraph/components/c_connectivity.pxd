from cugraph.structure.c_graph cimport *


cdef extern from "cugraph.h":

    cdef gdf_error gdf_connected_components(gdf_graph *graph, cugraph_cc_t connect_type, gdf_column *labels)

    ctypedef enum cugraph_cc_t:
        CUGRAPH_WEAK = 0,
        CUGRAPH_STRONG,
        NUM_CONNECTIVITY_TYPES
