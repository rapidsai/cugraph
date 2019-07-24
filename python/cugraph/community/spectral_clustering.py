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

from cugraph.community import spectral_clustering_wrapper


def spectralBalancedCutClustering(G,
                                  num_clusters,
                                  num_eigen_vects=2,
                                  evs_tolerance=.00001,
                                  evs_max_iter=100,
                                  kmean_tolerance=.00001,
                                  kmean_max_iter=100):
    """
    Compute a clustering/partitioning of the given graph using the spectral
    balanced cut method.

    Parameters
    ----------
    G : cuGraph.Graph
        cuGraph graph descriptor
    num_clusters : integer
         Specifies the number of clusters to find
    num_eigen_vects : integer
         Specifies the number of eigenvectors to use. Must be lower or equal to
         num_clusters.
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
    df : cudf.DataFrame
        GPU data frame containing two cudf.Series of size V: the vertex
        identifiers and the corresponding cluster assignments.

        df['vertex'] : cudf.Series
            contains the vertex identifiers
        df['cluster'] : cudf.Series
            contains the cluster assignments

    Examples
    --------
    >>> M = read_mtx_file(graph_file)
    >>> sources = cudf.Series(M.row)
    >>> destinations = cudf.Series(M.col)
    >>> G = cuGraph.Graph()
    >>> G.add_edge_list(sources,destinations,None)
    >>> DF = cuGraph.spectralBalancedCutClustering(G, 5)
    """

    df = spectral_clustering_wrapper.spectralBalancedCutClustering(
             G.graph_ptr,
             num_clusters,
             num_eigen_vects,
             evs_tolerance,
             evs_max_iter,
             kmean_tolerance,
             kmean_max_iter)

    return df


def spectralModularityMaximizationClustering(G,
                                             num_clusters,
                                             num_eigen_vects=2,
                                             evs_tolerance=.00001,
                                             evs_max_iter=100,
                                             kmean_tolerance=.00001,
                                             kmean_max_iter=100):
    """
    Compute a clustering/partitioning of the given graph using the spectral
    modularity maximization method.

    Parameters
    ----------
    G : cuGraph.Graph
        cuGraph graph descriptor
    num_clusters : integer
         Specifies the number of clusters to find
    num_eigen_vects : integer
         Specifies the number of eigenvectors to use. Must be lower or equal to
         num_clusters
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
    Clustering : cudf.DataFrame
        df['vertex'] : cudf.Series
            contains the vertex identifiers
        df['cluster'] : cudf.Series
            contains the cluster assignments

    Examples
    --------
    >>> M = read_mtx_file(graph_file)
    >>> sources = cudf.Series(M.row)
    >>> destinations = cudf.Series(M.col)
    >>> G = cuGraph.Graph()
    >>> G.add_edge_list(sources,destinations,None)
    >>> DF = cuGraph.spectralModularityMaximizationClustering(G, 5)
    """

    df = spectral_clustering_wrapper.spectralModularityMaximizationClustering(
             G.graph_ptr,
             num_clusters,
             num_eigen_vects,
             evs_tolerance,
             evs_max_iter,
             kmean_tolerance,
             kmean_max_iter)

    return df


def analyzeClustering_modularity(G, n_clusters, clustering):
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

    Examples
    --------
    >>> M = read_mtx_file(graph_file)
    >>> sources = cudf.Series(M.row)
    >>> destinations = cudf.Series(M.col)
    >>> G = cuGraph.Graph()
    >>> G.add_edge_list(sources,destinations,None)
    >>> DF = cuGraph.spectralBalancedCutClustering(G, 5)
    >>> score = cuGraph.analyzeClustering_modularity(G, 5, DF['cluster'])
    """

    score = spectral_clustering_wrapper.analyzeClustering_modularity(
                G.graph_ptr,
                n_clusters,
                clustering)

    return score


def analyzeClustering_edge_cut(G, n_clusters, clustering):
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

    Examples
    --------
    >>> M = read_mtx_file(graph_file)
    >>> sources = cudf.Series(M.row)
    >>> destinations = cudf.Series(M.col)
    >>> G = cuGraph.Graph()
    >>> G.add_edge_list(sources,destinations,None)
    >>> DF = cuGraph.spectralBalancedCutClustering(G, 5)
    >>> score = cuGraph.analyzeClustering_edge_cut(G, 5, DF['cluster'])
    """

    score = spectral_clustering_wrapper.analyzeClustering_edge_cut(
                G.graph_ptr,
                n_clusters,
                clustering)

    return score


def analyzeClustering_ratio_cut(G, n_clusters, clustering):
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

    Examples
    --------
    >>> M = read_mtx_file(graph_file)
    >>> sources = cudf.Series(M.row)
    >>> destinations = cudf.Series(M.col)
    >>> G = cuGraph.Graph()
    >>> G.add_edge_list(sources,destinations,None)
    >>> DF = cuGraph.spectralBalancedCutClustering(G, 5)
    >>> score = cuGraph.analyzeClustering_ratio_cut(G, 5, DF['cluster'])
    """

    score = spectral_clustering_wrapper.analyzeClustering_ratio_cut(
                G.graph_ptr,
                n_clusters,
                clustering)

    return score
