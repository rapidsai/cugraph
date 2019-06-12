from c_graph cimport *
from libcpp cimport bool

cdef extern from "cugraph.h":

    cdef gdf_error gdf_multi_pagerank (const size_t global_v, gdf_column *src_ptrs, gdf_column *dest_ptrs, gdf_column *pr, const float damping_factor, const int max_iter)
