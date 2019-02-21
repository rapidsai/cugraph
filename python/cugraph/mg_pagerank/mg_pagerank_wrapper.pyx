from libc.stdint cimport uintptr_t
from c_mg_pagerank cimport *

def mg_pagerank(input_df, pr_col_length, global_v):
    source_col = input_df[input_df.columns[0]]
    dest_col = input_df[input_df.columns[1]]
    pagerank_col = cudf.Series(np.ones(pr_col_length, dtype=np.float32))
    vertices_col = cudf.Series(np.ones(pr_col_length, dtype=np.int32))

    cdef uintptr_t source=create_column(source_col)
    cdef uintptr_t dest=create_column(dest_col)
    cdef uintptr_t pr_ptr = create_column(pagerank_col)
    cdef uintptr_t vid_ptr = create_column(vertices_col)


    gdf_multi_pagerank(<const size_t>global_v,
                    <gdf_column*>source,
                    <gdf_column*>dest,
                    <gdf_column*>vid_ptr,
                    <gdf_column*>pr_ptr,
                    <float> 0.85,#damping_factor,
                    <int> 20 #max_iter
                    )

    pr_df = cudf.DataFrame()
    pr_df['vertex'] = vertices_col
    pr_df['pagerank'] = pagerank_col
    return pr_df
