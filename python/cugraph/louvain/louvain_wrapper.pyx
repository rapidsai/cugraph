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
    input_graph : cuGraph.Graph
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
    >>> M = read_mtx_file(graph_file)
    >>> sources = cudf.Series(M.row)
    >>> destinations = cudf.Series(M.col)
    >>> G = cuGraph.Graph()
    >>> G.add_edge_list(sources,destinations,None)
    >>> louvain_parts, modularity_score = cuGraph.louvain(G)
    """

    cdef uintptr_t graph = input_graph.graph_ptr
    cdef gdf_graph* g = <gdf_graph*>graph

    err = gdf_add_adj_list(g)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

    num_vert = input_graph.num_vertices()

    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_vert, dtype=np.int32))
    cdef gdf_column c_index_col = get_gdf_column_view(df['vertex'])
    err = g.adjList.get_vertex_identifiers(&c_index_col)
    cudf.bindings.cudf_cpp.check_gdf_error(err)
    
    df['partition'] = cudf.Series(np.zeros(num_vert,dtype=np.int32))
    cdef gdf_column c_louvain_parts_col = get_gdf_column_view(df['partition'])

    cdef bool single_precision = False
    # this implementation is tied to cugraph.cu line 503
    # cudaDataType_t val_type = graph->adjList->edge_data?
    #     gdf_to_cudadtype(graph->adjList->edge_data): CUDA_R_32F;
    # this is tied to the low-level implementation detail of the lower level
    # function, and very vulnerable to low level changes. Better be
    # reimplemented, but we are planning to eventually remove nvgraph, so I may
    # leave as is right at this moment.
    if g.adjList.edge_data:
        if g.adjList.edge_data.dtype == GDF_FLOAT32:
            single_precision = True;
    else:
        single_precision = True;

    cdef float final_modularity_single_precision = 1.0
    cdef double final_modularity_double_precision = 1.0
    cdef int num_level = 0
    cdef gdf_error error

    if single_precision:
        err = gdf_louvain(<gdf_graph*>g,
                          <void*>&final_modularity_single_precision,
                          <void*>&num_level, &c_louvain_parts_col)
    else:
        err = gdf_louvain(<gdf_graph*>g,
                          <void*>&final_modularity_double_precision,
                          <void*>&num_level, &c_louvain_parts_col)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

    if single_precision:
        return df, <double>final_modularity_single_precision
    else:
        return df, final_modularity_double_precision
