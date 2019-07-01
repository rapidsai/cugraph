from cugraph.structure.c_graph cimport *

cdef gdf_column get_gdf_column_view(col)
cdef gdf_column get_gdf_column_ptr(ipc_data_ptr, col_len)
