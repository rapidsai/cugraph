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

from c_pagerank cimport *
from c_graph cimport *
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
import cudf
from librmm_cffi import librmm as rmm
#from pygdf import Column
import numpy as np

cpdef pagerank(G,alpha=0.85, personalization=None, max_iter=100, tol=1.0e-5):
    """
    Find the PageRank vertex values for a graph. cuGraph computes an approximation of the Pagerank eigenvector using the power method. 
    The number of iterations depends on the properties of the network itself; it increases when the tolerance descreases and/or alpha increases toward the limiting value of 1.
    The user is free to use default values or to provide inputs for the initial guess, tolerance and maximum number of iterations.
    
    Parameters
    ----------
    graph : cuGraph.Graph                  
       cuGraph graph descriptor, should contain the connectivity information as an edge list (edge weights are not used for this algorithm). 
       The transposed adjacency list will be computed if not already present.
    alpha : float                  
       The damping factor alpha represents the probability to follow an outgoing edge, standard value is 0.85. 
       Thus, 1.0-alpha is the probability to “teleport” to a random vertex. Alpha should be greater than 0.0 and strictly lower than 1.0.
    personalization : cudf.Dataframe
       GPU Dataframe containing the personalizatoin information.
       personalization['vertex']: Subset of vertices of graph for personalization
       personalization['values']: Personalization values for vertices

    tolerance : float              
       Set the tolerance the approximation, this parameter should be a small magnitude value. 
       The lower the tolerance the better the approximation. If this value is 0.0f, cuGraph will use the default value which is 1.0E-5. 
       Setting too small a tolerance can lead to non-convergence due to numerical roundoff. Usually values between 0.01 and 0.00001 are acceptable.
    max_iter  : int                
       The maximum number of iterations before an answer is returned. This can be used to limit the execution time and do an early exit before the solver reaches the convergence tolerance. 
       If this value is lower or equal to 0 cuGraph will use the default value, which is 100.
    
    Returns
    -------
    PageRank : cudf.DataFrame
        GPU data frame containing two cudf.Series of size V: the vertex identifiers and the corresponding PageRank values.

    Examples
    --------
    >>> M = read_mtx_file(graph_file)
    >>> sources = cudf.Series(M.row)
    >>> destinations = cudf.Series(M.col)
    >>> G = cuGraph.Graph()
    >>> G.add_edge_list(sources,destinations,None)
    >>> pr = cuGraph.pagerank(G, alpha = 0.85, max_iter = 500, tol = 1.0e-05)
    """

    cdef uintptr_t graph = G.graph_ptr
    cdef gdf_graph* g = <gdf_graph*>graph

    err = gdf_add_transposed_adj_list(g)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

    num_verts = G.number_of_vertices()

    df = cudf.DataFrame()  
    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    cdef gdf_column c_identifier_col = get_gdf_column_view(df['vertex']) 
    df['pagerank'] = cudf.Series(np.zeros(num_verts, dtype=np.float32))
    cdef gdf_column c_pagerank_col = get_gdf_column_view(df['pagerank'])    

    cdef gdf_column c_pers_vtx
    cdef gdf_column c_pers_val

    err = g.transposedAdjList.get_vertex_identifiers(&c_identifier_col)
    cudf.bindings.cudf_cpp.check_gdf_error(err)
    if personalization is None:
        err = gdf_pagerank(g, &c_pagerank_col, <gdf_column*> NULL, <gdf_column*> NULL,
                <float> alpha, <float> tol, <int> max_iter, <bool> 0)
    else:
        c_pers_vtx = get_gdf_column_view(personalization['vertex'])
        c_pers_val = get_gdf_column_view(personalization['values'])
        err = gdf_pagerank(g, &c_pagerank_col, &c_pers_vtx, &c_pers_val,
                <float> alpha, <float> tol, <int> max_iter, <bool> 0)

    cudf.bindings.cudf_cpp.check_gdf_error(err)

    return df
