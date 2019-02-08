from c_graph cimport *
from libcpp cimport bool

cdef extern from "nvgraph_gdf.h":

    cdef gdf_error gdf_sssp_nvgraph(gdf_graph *gdf_G,
                                    const int *source_vert,
                                    gdf_column *sssp_distances)

