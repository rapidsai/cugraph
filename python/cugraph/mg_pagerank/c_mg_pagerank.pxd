from c_graph cimport *
from libcpp cimport bool

cdef extern from "cugraph.h":

    cdef gdf_error gdf_multi_pagerank (const size_t global_v, const gdf_column *src_indices, const gdf_column *dest_indices, 
	                         gdf_column *v_idx, gdf_column *pagerank, const float damping_factor, const int max_iter)
