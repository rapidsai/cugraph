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
from c_graph cimport * 
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
import cudf
from libgdf_cffi import libgdf
from librmm_cffi import librmm as rmm
import numpy as np
from numpy.core.numeric import result_type

gdf_to_np_dtypes = {GDF_INT32:np.int32, GDF_INT64:np.int64, GDF_FLOAT32:np.float32, GDF_FLOAT64:np.float64}

cpdef jaccard_w(input_graph, weights, first=None, second=None):
    """
    Compute the weighted Jaccard similarity between each pair of vertices 
    connected by an edge. Jaccard similarity is defined between two sets as 
    the ratio of the volume of their intersection divided by the volume of 
    their union. In the context of graphs, the neighborhood of a vertex is seen 
    as a set. The Jaccard similarity weight of each edge represents the strength 
    of connection between vertices based on the relative similarity of their 
    neighbors.

    Parameters
    ----------
    graph : cuGraph.Graph                 
      cuGraph graph descriptor, should contain the connectivity information as 
      an edge list (edge weights are not used for this algorithm). The adjacency 
      list will be computed if not already present.   

    vect_weights_ptr : vector (array) of weights of size n = |G| (#vertices)

    Returns
    -------
    df  : cudf.DataFrame
      GPU data frame of size E containing the Jaccard weights. The ordering is 
          relative to the adjacency list.
          
      df['source']: The source vertex ID
      df['destination']: The destination vertex ID
      df['jaccard_coeff']: The computed weighted Jaccard coefficient between 
          the source and destination vertices. 
    Examples
    --------
    >>> M = ReadMtxFile(graph_file)
    >>> sources = cudf.Series(M.row)
    >>> destinations = cudf.Series(M.col)
    >>> G = cuGraph.Graph()
    >>> G.add_edge_list(sources,destinations,None)
    >>> jaccard_weights = cuGraph.jaccard_w(G, weights)
    """

    cdef uintptr_t graph = input_graph.graph_ptr
    cdef gdf_graph * g = < gdf_graph *> graph
    
    err = gdf_add_adj_list(g)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

    cdef uintptr_t result_ptr
    cdef uintptr_t weight_ptr
    cdef uintptr_t first_ptr
    cdef uintptr_t second_ptr
    cdef uintptr_t indices_ptr

    if type(first) == cudf.dataframe.series.Series and type(second) == cudf.dataframe.series.Series:
        resultSize = len(first)
        result = cudf.Series(np.ones(resultSize, dtype=np.float32))
        result_ptr = create_column(result)
        weight_ptr = create_column(weights)
        first_ptr = create_column(first)
        second_ptr = create_column(second)
        err = gdf_jaccard_list(g,
                               < gdf_column *> weight_ptr,
                               < gdf_column *> first_ptr,
                               < gdf_column *> second_ptr,
                               < gdf_column *> result_ptr)
        cudf.bindings.cudf_cpp.check_gdf_error(err)
        df = cudf.DataFrame()
        df['source'] = first
        df['destination'] = second
        df['jaccard_coeff'] = result
        return df

    elif first is None and second is None:
        resultSize = g.adjList.indices.size
        result = cudf.Series(np.ones(resultSize, dtype=np.float32))
        result_ptr = create_column(result)
        weight_ptr = create_column(weights)

        err = gdf_jaccard(g, < gdf_column *> weight_ptr, < gdf_column *> result_ptr)
        cudf.bindings.cudf_cpp.check_gdf_error(err)

        dest_data = rmm.device_array_from_ptr(< uintptr_t > g.adjList.indices.data,
                                            nelem=resultSize,
                                            dtype=gdf_to_np_dtypes[g.adjList.indices.dtype])
        df = cudf.DataFrame()
        df['source'] = cudf.Series(np.zeros(resultSize, dtype=gdf_to_np_dtypes[g.adjList.indices.dtype]))
        indices_ptr = create_column(df['source']) 
        err = g.adjList.get_source_indices(< gdf_column *> indices_ptr);
        cudf.bindings.cudf_cpp.check_gdf_error(err)
        df['destination'] = cudf.Series(dest_data)
        df['jaccard_coeff'] = result

        return df

    raise ValueError("Specify first and second or neither")

