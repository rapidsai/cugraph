# Copyright (c) 2019, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from c_jaccard cimport *
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
import cudf
from libgdf_cffi import libgdf
from librmm_cffi import librmm as rmm
import numpy as np
from cython cimport floating

gdf_to_np_dtypes = {GDF_INT32:np.int32, GDF_INT64:np.int64, GDF_FLOAT32:np.float32, GDF_FLOAT64:np.float64}

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

    data_type = np.float32

    if g.adjList:
        e = g.adjList.indices.size
        if g.adjList.edge_data:
            data_type =  gdf_to_np_dtypes[g.adjList.edge_data.dtype]
    else:
        e = g.edgeList.src_indices.size
        if g.edgeList.edge_data:
            data_type =  gdf_to_np_dtypes[g.edgeList.edge_data.dtype]

    weight_j_col = cudf.Series(np.zeros(e,dtype=data_type))
    cdef void* c_gamma
    cdef float num_f = 1.0
    cdef double num_d = 1.0
    if data_type is np.float32:
        c_gamma = <float*>&num_f
    else:
        c_gamma = <double*>&num_d
    cdef uintptr_t weight_j_col_ptr = create_column(weight_j_col)

    err = gdf_jaccard(<gdf_graph*>g, <void*>c_gamma, <gdf_column*>NULL, <gdf_column*>weight_j_col_ptr)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

    dest_data = rmm.device_array_from_ptr(<uintptr_t>g.adjList.indices.data,
                                            nelem=e,
                                            dtype=gdf_to_np_dtypes[g.adjList.indices.dtype]
                                            )
    df = cudf.DataFrame()
    source_col = cudf.Series(np.zeros(e,dtype=gdf_to_np_dtypes[g.adjList.indices.dtype]))
    df['source'] = source_col
    cdef uintptr_t src_indices_ptr = create_column(df['source']) 
    err = g.adjList.get_source_indices(<gdf_column*>src_indices_ptr);
    cudf.bindings.cudf_cpp.check_gdf_error(err)
    df['destination'] = cudf.Series(dest_data)
    df['jaccard_coeff'] = weight_j_col

    return df
