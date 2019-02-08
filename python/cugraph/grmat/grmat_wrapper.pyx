from c_grmat cimport *
from c_graph cimport *
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
import cudf
from librmm_cffi import librmm as rmm
import numpy as np

cpdef grmat_gen(argv):
    cdef size_t vertices = 0
    cdef size_t edges = 0
    cdef gdf_column* c_source_col = <gdf_column*>malloc(sizeof(gdf_column))
    c_source_col.dtype = GDF_INT32
    c_source_col.valid = NULL
    c_source_col.null_count = 0
    cdef gdf_column* c_dest_col = <gdf_column*>malloc(sizeof(gdf_column))
    c_dest_col.dtype = GDF_INT32
    c_dest_col.valid = NULL
    c_dest_col.null_count = 0
    #cdef gdf_column* c_val_col = <gdf_column*>malloc(sizeof(gdf_column))
    argv_bytes = argv.encode()
    cdef char* c_argv = argv_bytes
    
    err = gdf_grmat_gen (<char*>c_argv, vertices, edges, <gdf_column*>c_source_col, <gdf_column*>c_dest_col, <gdf_column*>0)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

    col_size = c_source_col.size
    cdef uintptr_t src_col_data = <uintptr_t>c_source_col.data
    cdef uintptr_t dest_col_data = <uintptr_t>c_dest_col.data
    
    src_data = rmm.device_array_from_ptr(src_col_data,
                                     nelem=col_size,
                                     dtype=np.int32,
                                     finalizer=rmm._make_finalizer(src_col_data, 0))
    dest_data = rmm.device_array_from_ptr(dest_col_data,
                                     nelem=col_size,
                                     dtype=np.int32,
                                     finalizer=rmm._make_finalizer(dest_col_data, 0))
    return vertices, edges, cudf.Series(src_data), cudf.Series(dest_data)


