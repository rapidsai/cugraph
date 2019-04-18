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

from c_louvain cimport *
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
import cudf
from libgdf_cffi import libgdf
from librmm_cffi import librmm as rmm
import numpy as np

cpdef nvLouvain(input_graph):
    """
    Compute the modularity optimizing partition of the input graph using the Louvain heuristic

    Parameters
    ----------
    graph : cuGraph.Graph                 
      cuGraph graph descriptor, should contain the connectivity information as an edge list (edge weights are not used for this algorithm).
      The adjacency list will be computed if not already present.   

    Returns
    -------
    louvain_parts, modularity_score  : cudf.DataFrame
      louvain_parts: GPU data frame of size V containing two columns: the vertex id 
          and the partition id it is assigned to.
      modularity_score: a double value containing the modularity score of the partitioning
 
    Examples
    --------
    >>> M = ReadMtxFile(graph_file)
    >>> sources = cudf.Series(M.row)
    >>> destinations = cudf.Series(M.col)
    >>> G = cuGraph.Graph()
    >>> G.add_edge_list(sources,destinations,None)
    >>> louvain_parts = cuGraph.louvain(G)
    """

    cdef uintptr_t graph = input_graph.graph_ptr
    cdef gdf_graph* g = <gdf_graph*>graph

    err = gdf_add_adj_list(g)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

    n = g.adjList.offsets.size - 1

    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(n, dtype=np.int32))
    cdef uintptr_t identifier_ptr = create_column(df['vertex'])
    err = g.adjList.get_vertex_identifiers(<gdf_column*>identifier_ptr)
    cudf.bindings.cudf_cpp.check_gdf_error(err)
    
    df['partition'] = cudf.Series(np.zeros(n,dtype=np.int32))
    cdef uintptr_t louvain_parts_col_ptr = create_column(df['partition'])
    cdef double final_modularity = 1.0
    cdef int num_level

    err = gdf_louvain(<gdf_graph*>g, <void*>&final_modularity, <void*>&num_level, <gdf_column*>louvain_parts_col_ptr)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

    cdef double fm = final_modularity
    cdef float tmp = (<float*>(<void*>&final_modularity))[0]
    if g.adjList.edge_data:
        if g.adjList.edge_data.dtype == GDF_FLOAT32:
            fm = tmp
    else:
        fm = tmp
    return df, fm                                      
