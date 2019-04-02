from c_graph cimport * 
cdef extern from "cugraph.h":
    cdef gdf_error gdf_jaccard (gdf_graph * graph,
                                gdf_column * weights,
                                gdf_column * result)
    
    cdef gdf_error gdf_jaccard_list(gdf_graph * graph,
                                    gdf_column * weights,
                                    gdf_column * first,
                                    gdf_column * second,
                                    gdf_column * result)
