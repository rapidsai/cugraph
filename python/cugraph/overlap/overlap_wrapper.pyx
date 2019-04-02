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
from cython cimport floating

cpdef overlap(input_graph, first=None, second=None):
    """
    Compute the Overlap Coefficient between each pair of vertices connected by an edge,
    or between arbitrary pairs of vertices specified by the user. Overlap Coefficient 
    is defined between two sets as the ratio of the volume of their intersection divided 
    by the smaller of their two volumes. In the context of graphs, the neighborhood of a vertex 
    is seen as a set. The Overlap Coefficient weight of each edge represents the strength 
    of connection between vertices based on the relative similarity of their neighbors.
    If first is specified but second is not, or vice versa, an exception will be thrown.

    Parameters
    ----------
    graph : cuGraph.Graph                 
      cuGraph graph descriptor, should contain the connectivity information as an edge list 
      (edge weights are not used for this algorithm).
      The adjacency list will be computed if not already present. 
    
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
      containing the Jaccard weights. The ordering is relative to the adjacency list, or that
      given by the specified vertex pairs.
      
      df['source']: The source vertex ID (will be identical to first if specified)
      df['destination']: The destination vertex ID (will be identical to second if specified)
      df['overlap_coeff']: The computed Jaccard coefficient between the source and destination
        vertices
 
    Examples
    --------
    >>> M = ReadMtxFile(graph_file)
    >>> sources = cudf.Series(M.row)
    >>> destinations = cudf.Series(M.col)
    >>> G = cuGraph.Graph()
    >>> G.add_edge_list(sources,destinations,None)
    >>> df = cugraph.overlap(G)
    """
    cdef uintptr_t graph = input_graph.graph_ptr
    cdef gdf_graph * g = < gdf_graph *> graph

    err = gdf_add_adj_list(< gdf_graph *> graph)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

    cdef uintptr_t result_ptr
    cdef uintptr_t first_ptr
    cdef uintptr_t second_ptr
    cdef uintptr_t src_indices_ptr

    if type(first) == cudf.dataframe.series.Series and type(second) == cudf.dataframe.series.Series:
        resultSize = len(first)
        result = cudf.Series(np.ones(resultSize, dtype=np.float32))
        result_ptr = create_column(result)
        first_ptr = create_column(first)
        second_ptr = create_column(second)
        err = gdf_overlap_list(g,
                               < gdf_column *> NULL,
                               < gdf_column *> first_ptr,
                               < gdf_column *> second_ptr,
                               < gdf_column *> result_ptr)
        cudf.bindings.cudf_cpp.check_gdf_error(err)
        df = cudf.DataFrame()
        df['source'] = first
        df['destination'] = second
        df['overlap_coeff'] = result
        return df

    elif first is None and second is None:
        e = g.adjList.indices.size
        result = cudf.Series(np.ones(e, dtype=np.float32), nan_as_null=False)
        result_ptr = create_column(result)

        err = gdf_overlap(g, < gdf_column *> NULL, < gdf_column *> result_ptr)
        cudf.bindings.cudf_cpp.check_gdf_error(err)

        dest_data = rmm.device_array_from_ptr(< uintptr_t > g.adjList.indices.data,
                                            nelem=e,
                                            dtype=gdf_to_np_dtypes[g.adjList.indices.dtype])
        df = cudf.DataFrame()
        df['source'] = cudf.Series(np.zeros(e, dtype=gdf_to_np_dtypes[g.adjList.indices.dtype]))
        src_indices_ptr = create_column(df['source']) 
        err = g.adjList.get_source_indices(< gdf_column *> src_indices_ptr);
        cudf.bindings.cudf_cpp.check_gdf_error(err)
        df['destination'] = cudf.Series(dest_data)
        df['overlap_coeff'] = result

        return df
    
    raise ValueError("Specify first and second or neither")
