from c_graph cimport *
from libcpp cimport cugraph_cc_t

cdef extern from "cugraph.h":

    cdef gdf_error gdf_connected_components(gdf_graph *graph, cugraph_cc_t connect_type, gdf_column *labels)
