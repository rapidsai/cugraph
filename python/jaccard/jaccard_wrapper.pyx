from c_jaccard cimport *
#from c_graph cimport *
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
import cudf
from libgdf_cffi import libgdf
from librmm_cffi import librmm as rmm
import numpy as np

gdf_to_np_dtypes = {GDF_INT32:np.int32, GDF_INT64:np.int64, GDF_FLOAT32:np.float32, GDF_FLOAT64:np.float64}

def _get_ctype_ptr(obj):
    # The manner to access the pointers in the gdf's might change, so
    # encapsulating access in the following 3 methods. They might also be
    # part of future gdf versions.
    return obj.device_ctypes_pointer.value

def _get_column_data_ptr(obj):
    return _get_ctype_ptr(obj._column._data.to_gpu_array())

def _get_column_valid_ptr(obj):
    return _get_ctype_ptr(obj._column._mask.to_gpu_array())


from enum import Enum
class cudaDataType(Enum):

    CUDA_R_16F= 2
    CUDA_C_16F= 6
    CUDA_R_32F= 0
    CUDA_C_32F= 4
    CUDA_R_64F= 1
    CUDA_C_64F= 5
    CUDA_R_8I = 3
    CUDA_C_8I = 7
    CUDA_R_8U = 8
    CUDA_C_8U = 9
    CUDA_R_32I= 10
    CUDA_C_32I= 11
    CUDA_R_32U= 12
    CUDA_C_32U= 13

np_to_cudaDataType = {np.int8:cudaDataType.CUDA_R_8I, np.int32:cudaDataType.CUDA_R_32I, np.float32:cudaDataType.CUDA_R_32F, np.float64:cudaDataType.CUDA_R_64F}
gdf_to_cudaDataType = {libgdf.GDF_INT8:cudaDataType.CUDA_R_8I, libgdf.GDF_INT32:cudaDataType.CUDA_R_32I, libgdf.GDF_FLOAT32:cudaDataType.CUDA_R_32F, libgdf.GDF_FLOAT64:cudaDataType.CUDA_R_64F}

cpdef nvJaccard(input_graph):
    """
    Compute the Jaccard similarity between each pair of vertices connected by an edge. Jaccard similarity is defined between two sets as the ratio of the volume of their intersection divided by the volume of their union. In the context of graphs, the neighborhood of a vertex is seen as a set. The Jaccard similarity weight of each edge represents the strength of connection between vertices based on the relative similarity of their neighbors.

    Parameters
    ----------
    graph : cuGraph.Graph                 
      cuGraph graph descriptor, should contain the connectivity information as an edge list (edge weights are not used for this algorithm).
      The adjacency list will be computed if not already present.   

    Returns
    -------
    jaccard_weights  : cudf.Serie
      GPU data frame of size E containing the Jaccard weights. The ordering is relative to the adjacency list.
 
    Examples
    --------
    >>> M = ReadMtxFile(graph_file)
    >>> sources = cudf.Series(M.row)
    >>> destinations = cudf.Series(M.col)
    >>> G = cuGraph.Graph()
    >>> G.add_edge_list(sources,destinations,None)
    >>> jaccard_weights = cuGraph.jaccard(G)
    """

    cdef uintptr_t graph = input_graph.graph_ptr
    cdef gdf_graph* g = <gdf_graph*>graph

    cdef uintptr_t adjList_ptr = <uintptr_t>g.adjList
    if adjList_ptr is 0:
        err = gdf_add_adj_list(<gdf_graph*>graph)
        cudf.bindings.cudf_cpp.check_gdf_error(err)

    cdef uintptr_t offsets_ptr = <uintptr_t>g.adjList.offsets.data
    cdef uintptr_t indices_ptr = <uintptr_t>g.adjList.indices.data
    cdef uintptr_t edge_value_ptr = <uintptr_t>g.adjList.edge_data
    cdef uintptr_t value_ptr
    val_type = gdf_to_cudaDataType[libgdf.GDF_FLOAT32].value
    if edge_value_ptr:
        value_ptr = <uintptr_t>g.adjList.edge_data.data
        val_type = gdf_to_cudaDataType[g.adjList.edge_data.dtype].value
    else:
        value_ptr = <uintptr_t>NULL

    n = g.adjList.offsets.size - 1
    e = g.adjList.indices.size
    index_type = gdf_to_cudaDataType[g.adjList.indices.dtype].value

    weight_j = cudf.Series(np.ones(e,dtype=np.float32), nan_as_null=False)
    cdef uintptr_t weight_j_ptr = _get_column_data_ptr(weight_j)
    cdef float c_gamma = 1.0

    nvgraphJaccard(<cudaDataType_t>index_type,
                   <cudaDataType_t>val_type,
                   <size_t>n,
                   <size_t>e,
                   <void*>offsets_ptr,
                   <void*>indices_ptr,
                   <void*>value_ptr,
                   <int>0,
                   <void*>NULL,
                   <void*>&c_gamma,
                   <void*>weight_j_ptr
                   )

    dest_data = rmm.device_array_from_ptr(<uintptr_t>g.adjList.indices.data,
                                            nelem=e,
                                            dtype=gdf_to_np_dtypes[g.adjList.indices.dtype],
                                            )
    df = cudf.DataFrame()
    df['source'] = cudf.Series(np.zeros(e,dtype=gdf_to_np_dtypes[g.adjList.indices.dtype]))
    cdef uintptr_t src_indices_ptr = create_column(df['source']) 
    err = g.adjList.get_source_indices(<gdf_column*>src_indices_ptr);
    cudf.bindings.cudf_cpp.check_gdf_error(err)
    df['destination'] = cudf.Series(dest_data)
    df['jaccard_coeff'] = weight_j

    return df
