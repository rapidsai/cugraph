from cugraph.structure.c_graph cimport *
from libcpp cimport bool


cdef extern from "cugraph.h":

    cdef gdf_error gdf_pagerank(gdf_graph *graph, gdf_column *pagerank,
            gdf_column *personalization_subset, gdf_column *personalization_values,
            float alpha, float tolerance, int max_iter, bool has_guess)

