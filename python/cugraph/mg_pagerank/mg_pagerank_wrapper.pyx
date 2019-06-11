from libc.stdint cimport uintptr_t
from c_mg_pagerank cimport *
import cudf
import numpy as np

def mg_pagerank(src_ptrs_info, dest_ptrs_info):
    cdef gdf_column* src_column_ptr = <gdf_column*>malloc(len(src_ptrs_info) * sizeof(gdf_column))
    cdef gdf_column* dest_column_ptr = <gdf_column*>malloc(len(dest_ptrs_info) * sizeof(gdf_column))
    
    n_gpus = len(src_ptrs_info);
    for i in range(n_gpus):
        src_column_ptr[i] = get_gdf_column_ptr(src_ptrs_info[i]["data"][0], src_ptrs_info[i]["shape"][0])
        dest_column_ptr[i] = get_gdf_column_ptr(dest_ptrs_info[i]["data"][0], dest_ptrs_info[i]["shape"][0])

    cdef gdf_column* pr_ptr = <gdf_column*>malloc(sizeof(gdf_column))
    gdf_multi_pagerank(<const size_t>n_gpus,
                     <gdf_column*> src_column_ptr,
                     <gdf_column*> dest_column_ptr,
                     <gdf_column*> pr_ptr,
                     <float> 0.85,#damping_factor,
                     <int> 20 #max_iter
                     )

    data = rmm.device_array_from_ptr(<uintptr_t> pr_ptr.data,
                                            nelem=pr_ptr.size,
                                            dtype=np.int32)
    x = cudf.DataFrame()
    x['pagerank'] = cudf.Series(data)
    return x
