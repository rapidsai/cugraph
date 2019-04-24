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

from c_overlap cimport * 
from c_graph cimport * 
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
import cudf
from libgdf_cffi import libgdf
from librmm_cffi import librmm as rmm
import numpy as np
from numpy.core.numeric import result_type

gdf_to_np_dtypes = {GDF_INT32:np.int32, GDF_INT64:np.int64, GDF_FLOAT32:np.float32, GDF_FLOAT64:np.float64}

cpdef overlap_w(input_graph, weights, first=None, second=None):
    """
    Compute the weighted Overlap Coefficient between each pair of vertices 
    connected by an edge, or between arbitrary pairs of vertices specified by 
    the user. Overlap Coefficient is defined between two sets as the ratio of the 
    volume of their intersection divided by the smaller of their volumes. In the 
    context of graphs, the neighborhood of a vertex is seen as a set. The 
    Overlap Coefficient weight of each edge represents the strength of connection 
    between vertices based on the relative similarity of their neighbors.
    If first is specified but second is not, or vice versa, an exception will be thrown.

    Parameters
    ----------
    graph : cuGraph.Graph                 
      cuGraph graph descriptor, should contain the connectivity information as 
      an edge list (edge weights are not used for this algorithm). The adjacency 
      list will be computed if not already present. 
      
    weights : cudf.Series
      Specifies the weights to be used for each vertex.
      
    first : cudf.Series
      Specifies the first vertices of each pair of vertices to compute for, must be specified
      along with second.
      
    second : cudf.Series
      Specifies the second vertices of each pair of vertices to compute for, must be specified
      along with first.

    Returns
    -------
    df  : cudf.DataFrame
      GPU data frame of size E (the default) or the size of the given pairs (first, second) 
      containing the overlap coefficients. The ordering is relative to the adjacency list, or that
      given by the specified vertex pairs.
          
      df['source']: The source vertex ID
      df['destination']: The destination vertex ID
      df['overlap_coeff']: The computed weighted Overlap coefficient between 
          the source and destination vertices. 
    Examples
    --------
    >>> M = read_mtx_file(graph_file)
    >>> sources = cudf.Series(M.row)
    >>> destinations = cudf.Series(M.col)
    >>> G = cuGraph.Graph()
    >>> G.add_edge_list(sources,destinations,None)
    >>> overlap = cugraph.overlap_w(G, weights)
    """

    cdef uintptr_t graph = input_graph.graph_ptr
    cdef gdf_graph * g = <gdf_graph*> graph
    
    err = gdf_add_adj_list(g)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

    cdef gdf_column c_result_col
    cdef gdf_column c_weight_col
    cdef gdf_column c_first_col
    cdef gdf_column c_second_col
    cdef gdf_column c_index_col

    if type(first) == cudf.dataframe.series.Series and type(second) == cudf.dataframe.series.Series:
        result_size = len(first)
        result = cudf.Series(np.ones(result_size, dtype=np.float32))
        c_result_col = get_gdf_column_view(result)
        c_weight_col = get_gdf_column_view(weights)
        c_first_col = get_gdf_column_view(first)
        c_second_col = get_gdf_column_view(second)
        err = gdf_overlap_list(g,
                               &c_weight_col,
                               &c_first_col,
                               &c_second_col,
                               &c_result_col)
        cudf.bindings.cudf_cpp.check_gdf_error(err)
        df = cudf.DataFrame()
        df['source'] = first
        df['destination'] = second
        df['overlap_coeff'] = result
        return df

    elif first is None and second is None:
        num_edges = input_graph.number_of_edges()
        result = cudf.Series(np.ones(num_edges, dtype=np.float32))
        c_result_col = get_gdf_column_view(result)
        c_weight_col = get_gdf_column_view(weights)

        err = gdf_overlap(g, &c_weight_col, &c_result_col)
        cudf.bindings.cudf_cpp.check_gdf_error(err)

        dest_data = rmm.device_array_from_ptr(<uintptr_t> g.adjList.indices.data,
                                            nelem=num_edges,
                                            dtype=gdf_to_np_dtypes[g.adjList.indices.dtype])
        df = cudf.DataFrame()
        df['source'] = cudf.Series(np.zeros(num_edges, dtype=gdf_to_np_dtypes[g.adjList.indices.dtype]))
        c_index_col = get_gdf_column_view(df['source']) 
        err = g.adjList.get_source_indices(&c_index_col);
        cudf.bindings.cudf_cpp.check_gdf_error(err)
        df['destination'] = cudf.Series(dest_data)
        df['overlap_coeff'] = result

        return df

    raise ValueError("Specify first and second or neither")
