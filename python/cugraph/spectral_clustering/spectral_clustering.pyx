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

from c_nvgraph cimport * 
from c_graph cimport * 
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
from libc.float cimport FLT_MAX_EXP
import cudf
from librmm_cffi import librmm as rmm
import numpy as np

cpdef spectralBalancedCutClustering(G,
                                    num_clusters,
                                    num_eigen_vects=2,
                                    evs_tolerance=.00001,
                                    evs_max_iter=100,
                                    kmean_tolerance=.00001,
                                    kmean_max_iter=100):
    """
    Compute a clustering/partitioning of the given graph using the spectral balanced
    cut method.
    
    Parameters
    ----------
    G : cuGraph.Graph                  
       cuGraph graph descriptor
    num_clusters : integer
        Specifies the number of clusters to find
    num_eigen_vects : integer
        Specifies the number of eigenvectors to use. Must be lower or equal to num_clusters.
    evs_tolerance: float
        Specifies the tolerance to use in the eigensolver
    evs_max_iter: integer
        Specifies the maximum number of iterations for the eigensolver
    kmean_tolerance: float
        Specifies the tolerance to use in the k-means solver
    kmean_max_iter: integer
        Specifies the maximum number of iterations for the k-means solver
    
    Returns
    -------
    DF : GPU data frame containing two cudf.Series of size V: the vertex identifiers and the corresponding SSSP distances.
        DF['vertex'] contains the vertex identifiers
        DF['cluster'] contains the cluster assignments
        
    Example:
    --------
    >>> M = ReadMtxFile(graph_file)
    >>> sources = cudf.Series(M.row)
    >>> destinations = cudf.Series(M.col)
    >>> G = cuGraph.Graph()
    >>> G.add_edge_list(sources,destinations,None)
    >>> DF = cuGraph.spectralBalancedCutClustering(G, 5)
    """

    cdef uintptr_t graph = G.graph_ptr
    cdef gdf_graph * g = < gdf_graph *> graph

    num_vert = g.adjList.offsets.size - 1

    # Create the output dataframe
    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_vert, dtype=np.int32))
    cdef uintptr_t identifier_ptr = create_column(df['vertex'])
    df['cluster'] = cudf.Series(np.zeros(num_vert, dtype=np.int32))
    cdef uintptr_t cluster_ptr = create_column(df['cluster'])
    
    # Set the vertex identifiers
    err = g.adjList.get_vertex_identifiers(< gdf_column *> identifier_ptr)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

    err = gdf_balancedCutClustering_nvgraph(g,
                                            num_clusters,
                                            num_eigen_vects,
                                            evs_tolerance,
                                            evs_max_iter,
                                            kmean_tolerance,
                                            kmean_max_iter,
                                            < gdf_column *> cluster_ptr)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

    return df

cpdef spectralModularityMaximizationClustering(G,
                                               num_clusters,
                                               num_eigen_vects=2,
                                               evs_tolerance=.00001,
                                               evs_max_iter=100,
                                               kmean_tolerance=.00001,
                                               kmean_max_iter=100):
    """
    Compute a clustering/partitioning of the given graph using the spectral modularity
    maximization method.
    
    Parameters
    ----------
    G : cuGraph.Graph                  
       cuGraph graph descriptor
    num_clusters : integer
        Specifies the number of clusters to find
    num_eigen_vects : integer
        Specifies the number of eigenvectors to use. Must be lower or equal to num_clusters
    evs_tolerance: float
        Specifies the tolerance to use in the eigensolver
    evs_max_iter: integer
        Specifies the maximum number of iterations for the eigensolver
    kmean_tolerance: float
        Specifies the tolerance to use in the k-means solver
    kmean_max_iter: integer
        Specifies the maximum number of iterations for the k-means solver
    
    Returns
    -------
    DF : GPU data frame containing two cudf.Series of size V: the vertex identifiers and the corresponding SSSP distances.
        DF['vertex'] contains the vertex identifiers
        DF['cluster'] contains the cluster assignments
        
    Example:
    --------
    >>> M = ReadMtxFile(graph_file)
    >>> sources = cudf.Series(M.row)
    >>> destinations = cudf.Series(M.col)
    >>> G = cuGraph.Graph()
    >>> G.add_edge_list(sources,destinations,None)
    >>> DF = cuGraph.spectralModularityMaximizationClustering(G, 5)
    """

    cdef uintptr_t graph = G.graph_ptr
    cdef gdf_graph * g = < gdf_graph *> graph

    num_vert = g.adjList.offsets.size - 1

    # Create the output dataframe
    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_vert, dtype=np.int32))
    cdef uintptr_t identifier_ptr = create_column(df['vertex'])
    df['cluster'] = cudf.Series(np.zeros(num_vert, dtype=np.int32))
    cdef uintptr_t cluster_ptr = create_column(df['cluster'])
    
    # Set the vertex identifiers
    err = g.adjList.get_vertex_identifiers(< gdf_column *> identifier_ptr)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

    err = gdf_spectralModularityMaximization_nvgraph(g,
                                                     num_clusters,
                                                     num_eigen_vects,
                                                     evs_tolerance,
                                                     evs_max_iter,
                                                     kmean_tolerance,
                                                     kmean_max_iter,
                                                     < gdf_column *> cluster_ptr)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

    return df

cpdef analyzeClustering_modularity(G, n_clusters, clustering):
    """
    Compute the modularity score for a partitioning/clustering
    
    Parameters
    ----------
    G : cuGraph.Graph                  
       cuGraph graph descriptor
    n_clusters : integer
        Specifies the number of clusters in the given clustering
    clustering : cudf.Series
        The cluster assignment to analyze.
    Returns
    -------
    score : float
        The computed modularity score
        
    Example:
    --------
    >>> M = ReadMtxFile(graph_file)
    >>> sources = cudf.Series(M.row)
    >>> destinations = cudf.Series(M.col)
    >>> G = cuGraph.Graph()
    >>> G.add_edge_list(sources,destinations,None)
    >>> DF = cuGraph.spectralBalancedCutClustering(G, 5)
    >>> score = cuGraph.analyzeClustering_modularity(G, 5, DF['cluster'])
    """
    cdef uintptr_t graph = G.graph_ptr
    cdef gdf_graph * g = < gdf_graph *> graph
    cdef uintptr_t clustering_ptr = create_column(clustering)
    cdef float score
    err = gdf_AnalyzeClustering_modularity_nvgraph(g, n_clusters, <gdf_column*>clustering_ptr, &score)
    cudf.bindings.cudf_cpp.check_gdf_error(err)
    return score

cpdef analyzeClustering_edge_cut(G, n_clusters, clustering):
    """
    Compute the edge cut score for a partitioning/clustering
    
    Parameters
    ----------
    G : cuGraph.Graph                  
       cuGraph graph descriptor
    n_clusters : integer
        Specifies the number of clusters in the given clustering
    clustering : cudf.Series
        The cluster assignment to analyze.
    Returns
    -------
    score : float
        The computed edge cut score
        
    Example:
    --------
    >>> M = ReadMtxFile(graph_file)
    >>> sources = cudf.Series(M.row)
    >>> destinations = cudf.Series(M.col)
    >>> G = cuGraph.Graph()
    >>> G.add_edge_list(sources,destinations,None)
    >>> DF = cuGraph.spectralBalancedCutClustering(G, 5)
    >>> score = cuGraph.analyzeClustering_edge_cut(G, 5, DF['cluster'])
    """
    cdef uintptr_t graph = G.graph_ptr
    cdef gdf_graph * g = < gdf_graph *> graph
    cdef uintptr_t clustering_ptr = create_column(clustering)
    cdef float score
    err = gdf_AnalyzeClustering_edge_cut_nvgraph(g, n_clusters, <gdf_column*>clustering_ptr, &score)
    cudf.bindings.cudf_cpp.check_gdf_error(err)
    return score

cpdef analyzeClustering_ratio_cut(G, n_clusters, clustering):
    """
    Compute the ratio cut score for a partitioning/clustering
    
    Parameters
    ----------
    G : cuGraph.Graph                  
       cuGraph graph descriptor
    n_clusters : integer
        Specifies the number of clusters in the given clustering
    clustering : cudf.Series
        The cluster assignment to analyze.
    Returns
    -------
    score : float
        The computed ratio cut score
        
    Example:
    --------
    >>> M = ReadMtxFile(graph_file)
    >>> sources = cudf.Series(M.row)
    >>> destinations = cudf.Series(M.col)
    >>> G = cuGraph.Graph()
    >>> G.add_edge_list(sources,destinations,None)
    >>> DF = cuGraph.spectralBalancedCutClustering(G, 5)
    >>> score = cuGraph.analyzeClustering_ratio_cut(G, 5, DF['cluster'])
    """
    cdef uintptr_t graph = G.graph_ptr
    cdef gdf_graph * g = < gdf_graph *> graph
    cdef uintptr_t clustering_ptr = create_column(clustering)
    cdef float score
    err = gdf_AnalyzeClustering_ratio_cut_nvgraph(g, n_clusters, <gdf_column*>clustering_ptr, &score)
    cudf.bindings.cudf_cpp.check_gdf_error(err)
    return score
