from c_graph cimport *

cdef extern from "cugraph.h":

    cdef gdf_error gdf_grmat_gen(const char* argv, const size_t &vertices, const size_t &edges, gdf_column* src, gdf_column* dest, gdf_column* val)
