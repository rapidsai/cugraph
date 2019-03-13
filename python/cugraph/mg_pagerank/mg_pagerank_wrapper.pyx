from libc.stdint cimport uintptr_t
from c_mg_pagerank cimport *

def mg_pagerank(input_df, global_v):
    source_col = input_df[input_df.columns[0]]
    dest_col = input_df[input_df.columns[1]]
    #pagerank_col = cudf.Series(np.ones(pr_col_length, dtype=np.float32))
    #vertices_col = cudf.Series(np.ones(pr_col_length, dtype=np.int32))
    cdef gdf_column* vid_ptr= <gdf_column*>malloc(sizeof(gdf_column))
    cdef gdf_column* pr_ptr= <gdf_column*>malloc(sizeof(gdf_column))    

    cdef uintptr_t source=create_column(source_col)
    cdef uintptr_t dest=create_column(dest_col)
    #cdef uintptr_t pr_ptr = create_column(pagerank_col)
    #cdef uintptr_t vid_ptr = create_column(vertices_col)


    gdf_multi_pagerank(<const size_t>global_v,
                    <gdf_column*>source,
                    <gdf_column*>dest,
                    <gdf_column*>vid_ptr,
                    <gdf_column*>pr_ptr,
                    <float> 0.85,#damping_factor,
                    <int> 20 #max_iter
                    )

    cdef uintptr_t vid_ptr_data = <uintptr_t>vid_ptr.data
    cdef uintptr_t pr_ptr_data = <uintptr_t>pr_ptr.data
    vid_data = rmm.device_array_from_ptr(vid_ptr_data,
                                     nelem=vid_ptr.size,
                                     dtype=np.int32,
                                     finalizer=rmm._make_finalizer(vid_ptr_data, 0))
    pr_data = rmm.device_array_from_ptr(pr_ptr_data,
                                     nelem=pr_ptr.size,
                                     dtype=np.float32,
                                     finalizer=rmm._make_finalizer(pr_ptr_data, 0))
    pr_df = cudf.DataFrame()
    pr_df['vertex'] = vid_data
    pr_df['pagerank'] = pr_data

    free(vid_ptr)
    free(pr_ptr)

    return pr_df
