from c_graph cimport *
from libcpp cimport bool

cdef extern from "cugraph.h":

    cdef gdf_error gdf_snmg_pagerank (
            gdf_column **src_col_ptrs, 
            gdf_column **dest_col_ptrs, 
            gdf_column *pr_col, 
            const size_t n_gpus, 
            const float damping_factor, 
            const int n_iter)
