from cugraph.structure.utils_wrapper import *
from cugraph.opg.link_analysis cimport mg_pagerank as c_pagerank
import cudf
from cugraph.structure.graph_new cimport *
from cugraph.structure import graph_new_wrapper
from libc.stdint cimport uintptr_t


def mg_pagerank(input_df, alpha=0.85, personalization=None, max_iter=100, tol=1.0e-5, nstart=None):
    """
    Call pagerank
    """
    print("IN")

    [src, dst] = graph_new_wrapper.datatype_cast([input_df['src'], input_df['dst']], [np.int32])
    [weights] = graph_new_wrapper.datatype_cast([input_df['value']], [np.float32, np.float64])

    offsets, indices, weights = coo2csr(dst, src, weights)

    num_verts = 34 #FIXME Get global number of vertices
    num_edges = len(indices)

    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    df['pagerank'] = cudf.Series(np.zeros(num_verts, dtype=np.float32))

    cdef uintptr_t c_identifier = df['vertex'].__cuda_array_interface__['data'][0];
    cdef uintptr_t c_pagerank_val = df['pagerank'].__cuda_array_interface__['data'][0];

    cdef uintptr_t c_offsets = offsets.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_indices = indices.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights = <uintptr_t>NULL

    if weights is not None:
        c_weights = weights.__cuda_array_interface__['data'][0]

    cdef GraphCSC[int,int,float] graph_float
    cdef GraphCSC[int,int,double] graph_double

    print("BEFORE CPP CALL")
    if (df['pagerank'].dtype == np.float32): 
        graph_float = GraphCSC[int,int,float](<int*>c_offsets, <int*>c_indices, <float*>c_weights, num_verts, num_edges)

        c_pagerank.mg_pagerank_temp[int,int,float](graph_float, <float*> c_pagerank_val)

    else: 
        graph_double = GraphCSC[int,int,double](<int*>c_offsets, <int*>c_indices, <double*>c_weights, num_verts, num_edges)
        c_pagerank.mg_pagerank_temp[int,int,double](graph_double, <double*> c_pagerank_val)

    return df
