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

from c_sssp cimport *
from c_graph cimport *
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
from libc.float cimport FLT_MAX_EXP
import cudf
from librmm_cffi import librmm as rmm
#from pygdf import Column
import numpy as np

gdf_to_np_dtypes = {GDF_INT32:np.int32, GDF_INT64:np.int64, GDF_FLOAT32:np.float32, GDF_FLOAT64:np.float64}

cpdef sssp(G, source):
    """
    Compute the distance and predecessors for shortest paths from the specified source to all the vertices in the graph. The distances column will
    store the distance from the source to each vertex. The predecessors column will store each vertex's predecessor in the shortest path. Vertices
    that are unreachable will have a distance of infinity denoted by the maximum value of the data type and the predecessor set as -1. The source vertex's 
    predecessor is also set to -1. Graphs with negative weight cycles are not supported.
    
    Parameters
    ----------
    graph : cuGraph.Graph                  
       cuGraph graph descriptor with connectivity information. Edge weights, if present, should be single or double precision floating point values
    source : int                  
       Index of the source vertex

    Returns
    -------
    df : cudf.DataFrame
        df['vertex'][i] gives the vertex id of the i'th vertex
        df['distance'][i] gives the path distance for the i'th vertex from the starting vertex
        df['predecessor'][i] gives the vertex id of the vertex that was reached before the i'th vertex in the traversal
    
    Examples
    --------
    >>> M = read_mtx_file(graph_file)
    >>> sources = cudf.Series(M.row)
    >>> destinations = cudf.Series(M.col)
    >>> G = cuGraph.Graph()
    >>> G.add_edge_list(sources,destinations,None)
    >>> distances = cuGraph.sssp(G, source)
    """

    cdef uintptr_t graph = G.graph_ptr
    cdef gdf_graph* g = <gdf_graph*>graph

    err = gdf_add_adj_list(g)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

    num_verts = G.number_of_vertices()

    data_type = np.float32
    if g.adjList.edge_data:
        data_type = gdf_to_np_dtypes[g.adjList.edge_data.dtype]

    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    cdef gdf_column c_identifier_col = get_gdf_column_view(df['vertex'])
    df['distance'] = cudf.Series(np.zeros(num_verts, dtype=data_type))
    cdef gdf_column c_distance_col = get_gdf_column_view(df['distance'])
    df['predecessor'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    cdef gdf_column c_predecessors_col = get_gdf_column_view(df['predecessor'])

    err = g.adjList.get_vertex_identifiers(&c_identifier_col)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

    err = gdf_sssp(g, &c_distance_col, &c_predecessors_col, <int>source)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

    return df

def filter_unreachable(df):
    """
    Remove unreachable vertices from the result of SSSP or BFS
    
    Parameters
    ----------
    df : cudf.DataFrame that is the output of SSSP or BFS 

    Returns
    -------
    df : filtered cudf.DataFrame with only reachable vertices
        df['vertex'][i] gives the vertex id of the i'th vertex
        df['distance'][i] gives the path distance for the i'th vertex from the starting vertex
        df['predecessor'][i] gives the vertex that was reached before the i'th vertex in the traversal
    """
    if('distance' not in df):
        raise KeyError("No distance column found in input data frame")
    if(np.issubdtype(df['distance'].dtype, np.integer)):
        max_val = np.iinfo(df['distance'].dtype).max
        return df[df.distance != max_val]
    elif(np.issubdtype(df['distance'].dtype, np.inexact)):
        max_val = np.finfo(df['distance'].dtype).max
        return df[df.distance != max_val]
    else:
        raise TypeError("distace type unsupported") 
